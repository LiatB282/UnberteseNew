import sched
import torch
import utils
import random
import numpy as np
import os
import json
import pickle
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
from torch.nn import Softmin, CrossEntropyLoss, MSELoss
from transformers import AdamW, BertTokenizer, BertForMaskedLM, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import logging
from cli import parse_args
import wandb
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def init_optimizer(model, args, total_number_of_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                          correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = utils.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_number_of_steps
    )

    return optimizer, scheduler

def init_model(args, device):
    logger.info(f"Initializing pretrained model from {args.pretrained_model_path}")
    pretrained_model = BertForMaskedLM.from_pretrained(args.pretrained_model_path).to(device)    
    return pretrained_model

class Trainer():
    def __init__(self, args, model, optimizer, scheduler, tokenizer, train_loader, eval_loader, device):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        
        self.softmin = Softmin(dim=1)        
        self.mlm_bert = BertForMaskedLM.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir).to(device)
        self.mlm_bert.eval()
        self.freeze_model(self.mlm_bert)
        self.paraphrase_model = AutoModel.from_pretrained(args.paraphrase_model_name, cache_dir=args.cache_dir).to(device)
        self.paraphrase_model.eval()
        self.freeze_model(self.paraphrase_model)
        self.paraphrase_classifer_model = AutoModelForSequenceClassification.from_pretrained(args.paraphrase_classifer_model_name, cache_dir=args.cache_dir).to(device)
        self.paraphrase_classifer_model.eval()
        self.freeze_model(self.paraphrase_classifer_model)

        self.sim_cse_tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_model_name, cache_dir=args.cache_dir)
        self.vocab_size = self.mlm_bert.config.vocab_size
        #self.mask_embedding = self.mlm_bert.bert(torch.tensor([tokenizer.encode("[MASK]", add_special_tokens=False)], device=device))[0].flatten()
        self.mask_embedding = self.mlm_bert.bert.embeddings.word_embeddings.weight[103]
        self.one_tensor = torch.tensor(1.0, device=self.device)
        self.zero_tensor = torch.tensor(0.0, device=self.device)
        vocab_indices = torch.tensor(list(range(self.vocab_size)), device=device)
        self.vocab_embeddings = self.mlm_bert.bert.embeddings.word_embeddings(vocab_indices)
        #self.beta = torch.nn.Parameter(torch.randn(1)).to(device)
        #self.index = faiss.IndexFlatL2(self.vocab_embeddings.shape[-1])
        #self.index.add(self.vocab_embeddings.cpu().numpy())

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def evaluate(self, epoch_number):
        logger.info(f"Evaluting epoch: {epoch_number}")

        self.model.eval()
        eval_accuracy = 0
        steps = 0
        loss = 0
        ce_loss = 0
        mask_loss = 0
        vocab_loss = 0
        paraphrase_loss = 0
        step = 0 
        for batch in tqdm(self.eval_loader):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids = batch[0]
            mask_label_tensor = batch[1]
            attention_mask = input_ids != self.tokenizer.pad_token_id

            with torch.no_grad():
                outputs = self.model.bert(input_ids=input_ids)

            last_hidden_states = outputs.last_hidden_state

            # if step % 100 == 0:
            #     self.print_rewrites(input_ids, last_hidden_states)

            # loss
            batch_loss, batch_ce_loss, batch_mask_loss, batch_vocab_loss, batch_paraphrase_loss, batch_accuracy = self.calc_loss_and_accuracy(last_hidden_states, mask_label_tensor, input_ids, attention_mask)
            loss += batch_loss
            ce_loss += batch_ce_loss
            mask_loss += batch_mask_loss
            vocab_loss += batch_vocab_loss
            eval_accuracy += batch_accuracy
            paraphrase_loss += batch_paraphrase_loss
            steps = steps + 1

        eval_accuracy = eval_accuracy / steps
        loss = loss / steps
        ce_loss = ce_loss / steps
        mask_loss = mask_loss / steps
        vocab_loss = vocab_loss / steps
        paraphrase_loss = paraphrase_loss / steps
        
        logger.info(f"Eval accuracy: {eval_accuracy}, eval loss: {loss}")

        d = {"eval/accuracy": eval_accuracy, 
            "eval/loss": loss, 
            "eval/ce_loss": ce_loss, 
            "eval/mask_loss": mask_loss, 
            "eval/vocab_loss": vocab_loss,
            "eval/paraphrase_loss": paraphrase_loss,
            "eval/epoch": epoch_number}

        if self.args.use_wandb:
            wandb.log(d)

        return eval_accuracy, loss

    def print_rewrites(self, input_ids, rewrite_embeds):
        rewrite_embeds = rewrite_embeds.detach().cpu().numpy()
        D, I = self.index.search(rewrite_embeds.reshape(-1, rewrite_embeds.shape[2]), 1)
        rewrited_sequences = I.reshape(rewrite_embeds.shape[0], -1)
        input_ids = input_ids.detach().cpu().numpy()

        for i in range(len(input_ids)):
            rewrited_sequence = utils.bert_decode_clean(rewrited_sequences[i], self.tokenizer)
            original_query = utils.bert_decode_clean(input_ids[i], self.tokenizer)
            logger.info(f"Original query: {original_query} | Rewrited query: {rewrited_sequence}")

    def calc_accuracy(self, mask_softmin_scores, mlm_bert_logits, mask_label_tensor):
        mask_indices = torch.argmax(mask_softmin_scores, dim=1).unsqueeze(1).repeat(1, mlm_bert_logits.shape[2]).unsqueeze(1)
        preds_logits = torch.gather(mlm_bert_logits, 1, mask_indices).detach().cpu().numpy().squeeze(1)

        sort_axis = 0 if len(preds_logits) == self.vocab_size else 1

        predictions = np.argmax(preds_logits, axis=sort_axis)
        label_ids = mask_label_tensor.squeeze(1).detach().cpu().numpy()
        if sort_axis == 0:
            predictions = np.array([predictions])

        text_predictions = []
        text_labels = []
        
        for i in range(len(predictions)):
            text_predictions.append(self.tokenizer.decode([predictions[i]], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            text_labels.append(self.tokenizer.decode([label_ids[i]], skip_special_tokens=False, clean_up_tokenization_spaces=True))
        
        accuracy = utils.single_token_simple_accuracy(text_predictions, text_labels)
        return accuracy

    def train_epoch(self, epoch_number):
        logger.info("Training epoch number: {}".format(epoch_number))

        epoch_iterator = tqdm(self.train_loader, desc="Epoch {} Iteration".format(epoch_number))
        self.model.train()

        rolling_loss = 0
        rolling_accuracy = 0
        rolling_steps = 100

        for step, batch in enumerate(epoch_iterator):
            #batch = tuple(t.to(self.device) for t in batch)
            input_ids = batch[0].to(self.device)
            mask_label_tensor = batch[1].to(self.device)
            attention_mask = input_ids != self.tokenizer.pad_token_id

            # Forward pass
            outputs = self.model.bert(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state

            # loss
            loss, ce_loss, mask_loss, vocab_loss, paraphrase_loss, accuracy = self.calc_loss_and_accuracy(last_hidden_states, mask_label_tensor, input_ids, attention_mask)
            rolling_loss += loss
            rolling_accuracy += accuracy

            if (step + 1) % rolling_steps == 0:
                #self.print_rewrites(input_ids, last_hidden_states)

                logger.info(f"Epoch {epoch_number} avg for {step} steps: train accuracy: {rolling_accuracy}, train loss: {rolling_loss}")
                rolling_loss = 0
                rolling_accuracy = 0

            d = {"train/accuracy": accuracy, 
                "train/loss": loss, 
                "train/ce_loss": ce_loss, 
                "train/mask_loss": mask_loss, 
                "train/vocab_loss": vocab_loss,
                "train/paraphrase_loss": paraphrase_loss,
                "train/epoch": epoch_number
                }
            
            if self.args.use_wandb:
                wandb.log(d)

            if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)


            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
            else:
                continue

    def calc_loss_and_accuracy(self, last_hidden_states, mask_label, input_ids, attention_mask):
        # mask_distances_scores = self.get_mask_distances_scores(last_hidden_states)
        # label_logits = self.get_mlm_bert_label_logits(last_hidden_states, mask_distances_scores)
        #mask_softmin_scores = self.softmin(self.beta*torch.norm(last_hidden_states - self.mask_embedding, dim=2))
        mask_softmin_scores = self.softmin(torch.norm(last_hidden_states - self.mask_embedding, dim=2))
        token_type_ids = torch.zeros_like(input_ids, device=self.device)
        mlm_bert_hidden_states = self.mlm_bert.bert(inputs_embeds=last_hidden_states, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        mlm_bert_logits = self.mlm_bert.cls(mlm_bert_hidden_states)

        # CE loss
        loss_func = CrossEntropyLoss(reduction='none')
        batch_size, seq_len, vocab_size = mlm_bert_logits.shape
        ce_loss = loss_func(mlm_bert_logits.view(-1, self.vocab_size), mask_label.repeat(1, seq_len).view(-1)).view(batch_size, seq_len)
        ce_loss = torch.sum(mask_softmin_scores * ce_loss, dim=1) # sum across sequence
        ce_loss = torch.mean(ce_loss) # mean across batch
        ce_loss = -1*ce_loss

        # mask loss
        # loss_func = MSELoss()
        # mask_loss = loss_func(torch.max(mask_distances_scores, dim=1)[0], self.one_tensor)
        mask_loss = -torch.max(mask_softmin_scores, dim=1)[0]
        mask_loss = torch.mean(mask_loss)

        # vocab loss
        loss_func = MSELoss()
        all_dist = utils.compte_all_dist(last_hidden_states, self.vocab_embeddings)
        vocab_loss = loss_func(torch.mean(torch.min(all_dist, dim=2)[0]), self.zero_tensor)

        # Paraphrase loss
        #paraphrase_loss = self.calc_paraphrase_classification_loss(last_hidden_states, input_ids)#self.calc_simsce_paraphrase_loss(last_hidden_states, input_ids)
        paraphrase_loss = self.calc_paraphrase_loss(last_hidden_states, input_ids, attention_mask)

        loss = (self.args.label_loss_weight * ce_loss)  \
                    + (self.args.mask_loss_weight * mask_loss) \
                    + (self.args.tokens_dist_loss_weight * vocab_loss) \
                    + (self.args.paraphrase_loss_weight * paraphrase_loss)
        loss = loss.mean()

        accuracy = self.calc_accuracy(mask_softmin_scores, mlm_bert_logits, mask_label)
                    
        return loss, ce_loss, mask_loss, vocab_loss, paraphrase_loss, accuracy

    def calc_paraphrase_loss(self, last_hidden_states, input_ids, attention_mask):
        return self.calc_simsce_paraphrase_loss(last_hidden_states, input_ids) #0.05*self.calc_simsce_paraphrase_loss(last_hidden_states, input_ids) + 0.95*self.calc_paraphrase_classification_loss(last_hidden_states, input_ids)

    def calc_mean_pooling_dist_loss(self, last_hidden_states, input_ids, attention_mask):
        loss_func = MSELoss()
        original_query_mean_pooling = utils.mean_pooling(self.mlm_bert.bert.embeddings.word_embeddings(input_ids), attention_mask)
        rewriter_query_mean_pooling = utils.mean_pooling(last_hidden_states, attention_mask)
        rewrite_distance = torch.norm(rewriter_query_mean_pooling - original_query_mean_pooling, dim=1)
        return loss_func(rewrite_distance, torch.zeros_like(rewrite_distance, device=self.device))

    def calc_mean_pooling_dot_product_loss(self, last_hidden_states, input_ids, attention_mask):
        loss_func = torch.nn.BCEWithLogitsLoss()
        original_query_mean_pooling = utils.mean_pooling(self.mlm_bert.bert.embeddings.word_embeddings(input_ids), attention_mask)
        rewriter_query_mean_pooling = utils.mean_pooling(last_hidden_states, attention_mask)
        rewrite_score = torch.matmul(original_query_mean_pooling, rewriter_query_mean_pooling.T)
        return loss_func(rewrite_score, torch.ones_like(rewrite_score, device=self.device))

    def calc_paraphrase_classification_loss(self, last_hidden_states, input_ids):
        embeds_list = []
        final_token_type_ids = []

        # To match the classifier input - merging the input ids and rewriter's embeddings to the static embeddings of both [embed1] [SEP] [embed2]
        for i in range(input_ids.shape[0]):
            current_input_ids = input_ids[i].unsqueeze(0)
            attention_mask = current_input_ids != self.tokenizer.pad_token_id
            # step 1: remove padding ids
            current_input_ids = torch.masked_select(current_input_ids, attention_mask)

            # step 2: get embeds
            input_embeds1 = self.mlm_bert.bert.embeddings.word_embeddings(current_input_ids).unsqueeze(0)

            mask_indices = torch.sum(attention_mask, dim=1)[0]
            input_embeds2 = last_hidden_states[i].unsqueeze(0)
            input_embeds2 = input_embeds2[:,1:mask_indices,:]

            embeds = torch.cat((input_embeds1, input_embeds2), 1).squeeze(0)
            embeds_list.append(embeds)

            token_type_ids = torch.arange(embeds.shape[0]) < input_embeds1.shape[1]
            final_token_type_ids.append(token_type_ids)


        final_token_type_ids = torch.nn.utils.rnn.pad_sequence(final_token_type_ids, batch_first=True).to(self.device)
        attention_mask = torch.zeros_like(final_token_type_ids, device=self.device) == 0
        final_token_type_ids = ~final_token_type_ids
        pad_embed = self.mlm_bert.bert.embeddings.word_embeddings.weight[self.tokenizer.pad_token_id]
        batch_size = final_token_type_ids.shape[0]
        max_len = final_token_type_ids.shape[1]
        final_embeds_list = []
        for i in range(batch_size):
            embeds = embeds_list[i]
            padding_number = max_len - embeds.shape[0] 
            if padding_number > 0:
                attention_mask[i, embeds.shape[0]:] = False
                pads_to_add = pad_embed.repeat(padding_number, 1)
                embeds = torch.cat((embeds, pads_to_add), dim=0)
            final_embeds_list.append(embeds.unsqueeze(0))

        final_embeds = torch.cat(final_embeds_list, dim=0)

        with torch.no_grad():
            logits = self.paraphrase_classifer_model(inputs_embeds=final_embeds, token_type_ids=final_token_type_ids.long(), attention_mask=attention_mask).logits

        loss_func = CrossEntropyLoss()
        target = torch.ones(logits.shape[0], device=self.device, dtype=torch.long)
        return loss_func(logits, target)


    def calc_simsce_paraphrase_loss(self, last_hidden_states, input_ids):
        attention_mask = input_ids != self.tokenizer.pad_token_id

        original_query_simcse_embed = self.paraphrase_model(input_ids, attention_mask=attention_mask, return_dict=True).pooler_output
        rewriter_query_simcse_embed = self.paraphrase_model(inputs_embeds=last_hidden_states, attention_mask=attention_mask, return_dict=True).pooler_output
        
        # # Cosine similarity 
        loss_func = torch.nn.BCEWithLogitsLoss()
        original_query_simcse_embed = torch.nn.functional.normalize(original_query_simcse_embed, dim=1)
        rewriter_query_simcse_embed = torch.nn.functional.normalize(rewriter_query_simcse_embed, dim=1)
        rewrite_cosine_similarity = torch.matmul(rewriter_query_simcse_embed, original_query_simcse_embed.T)
        paraphrase_loss = loss_func(rewrite_cosine_similarity, torch.ones_like(rewrite_cosine_similarity, device=self.device))

        return paraphrase_loss

    def _save_checkpoint(self, epoch: int, offset = 0) -> str:
        args = self.args
        model_to_save = utils.get_model_obj(self.model)
        cp = os.path.join(
            args.output_dir,
            args.checkpoint_file_name
            + "." + str(epoch)
            + ("." + str(offset) if offset > 0 else ""),
        )

        state = utils.CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            offset,
            epoch,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def train(self):
        logger.info("Training... ")
        best_accuracy, loss = self.evaluate(-1)
        logger.info(f"Best accuracy: {best_accuracy} loss: {loss}")

        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for epoch_number, e in enumerate(train_iterator):
            self.train_epoch(epoch_number)

            accuracy, loss = self.evaluate(epoch_number)
            if accuracy > best_accuracy:
                logger.info(f"New best accuracy: {accuracy} loss: {loss}")
            if not self.args.dont_save:
                self._save_checkpoint(epoch_number)

def load_lama_examples(args, tokenizer, examples, is_dev):
    logger.info("Loading to device {} examples".format(len(examples)))
    # Encode the input to the encoder (the question)
    input_ids_list = []
    label_id_list = []

    valid_examples_counter = 0
    for example in examples:
        label = example.label
        label_id = tokenizer.encode(label, add_special_tokens=False)

        if len(label_id) > 1:
            continue

        label_id_list.append(label_id)
        masked_sentence = example.masked_sentence

        input_ids = tokenizer.encode(masked_sentence,
                                        pad_to_max_length=True,
                                        max_length=args.max_seq_length,
                                        add_special_tokens=True)
        input_ids_list.append(input_ids)

        valid_examples_counter += 1

    assert (len(input_ids_list) > 0)

    # Convert inputs to PyTorch tensors
    input_ids_tensor = torch.tensor(input_ids_list)
    labels_tensor = torch.tensor(label_id_list)

    dataset = TensorDataset(input_ids_tensor, labels_tensor)

    if is_dev:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
    else:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size)


    logger.info("Loaded {0} examples out of {1}".format(valid_examples_counter, len(examples)))

    return dataloader, dataset


def init_wandb(model, wandb_name, config):    
    wandb_run = wandb.init(
            project="UNBERTESE",
            name=wandb_name,
            entity='liatbezalel',
            config=config
        )

    wandb.watch(model)

    return wandb_run



def main():
    args = parse_args()
    # if "JOB_NAME" in os.environ:
    #     wandb_name = os.environ["JOB_NAME"]
    # else:
    wandb_name = 'unertese' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

    args.output_dir = f"{args.output_dir}/{wandb_name}"
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        
    utils.print_args(args)

    device, n_gpu = utils.get_device()
    tokenizer = BertTokenizer.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir)

    train_dataloader, dev_dataloader, number_of_train_examples = utils.load_data(args, device, tokenizer)
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    num_train_steps = int(number_of_train_examples / train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    model = init_model(args, device)
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
        
    optimizer, scheduler = init_optimizer(model, args, num_train_steps)
    trainer = Trainer(args, model, optimizer, scheduler, tokenizer, train_dataloader, dev_dataloader, device)
    
    if args.is_debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.use_wandb:
        wandb_run = init_wandb(model, wandb_name, vars(args))

    trainer.train()

    if args.use_wandb:
        wandb_run.finish()

main()