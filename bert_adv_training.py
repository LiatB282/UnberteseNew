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
from transformers import AdamW, BertTokenizer, BertForMaskedLM, AutoModel, AutoTokenizer
import logging
from cli import parse_adv_args
import wandb

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
    logger.info(f"Initializing pretrained model {args.mlm_bert_model_name}")
    pretrained_model = BertForMaskedLM.from_pretrained(args.mlm_bert_model_name).to(device)    
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

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def evaluate(self, epoch_number):
        logger.info(f"Evaluting epoch: {epoch_number}")

        self.model.eval()
        eval_accuracy = 0
        steps = 0
        loss = 0
        step = 0 
        for batch in tqdm(self.eval_loader):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids = batch[0]
            mask_label_tensor = batch[1]
            attention_mask = input_ids != self.tokenizer.pad_token_id
            token_type_ids = torch.zeros_like(attention_mask, dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

            # loss
            batch_loss, batch_accuracy = self.calc_loss_and_accuracy(logits, mask_label_tensor, input_ids)
            loss += batch_loss
            eval_accuracy += batch_accuracy
            steps = steps + 1

        eval_accuracy = eval_accuracy / steps
        loss = loss / steps
        
        logger.info(f"Eval accuracy: {eval_accuracy}, eval loss: {loss}")

        d = {"adv-eval/accuracy": eval_accuracy, 
            "adv-eval/loss": loss, 
            "adv-eval/epoch": epoch_number}

        if self.args.use_wandb:
            wandb.log(d)

        return eval_accuracy, loss

    def calc_accuracy(self, logits, mask_label_tensor, input_ids):
        mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_logits = logits[mask_indices].detach().cpu().numpy()
        predictions = np.argmax(mask_logits, axis=1)
        label_ids = mask_label_tensor.squeeze(1).detach().cpu().numpy()

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
            attention_mask = input_ids != self.tokenizer.pad_token_id
            token_type_ids = torch.zeros_like(attention_mask, dtype=torch.long, device=self.device)
            mask_label_tensor = batch[1].to(self.device)

            # Forward pass
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            # loss
            loss, accuracy = self.calc_loss_and_accuracy(logits, mask_label_tensor, input_ids)
            rolling_loss += loss
            rolling_accuracy += accuracy

            if (step + 1) % rolling_steps == 0:
                #self.print_rewrites(input_ids, last_hidden_states)

                logger.info(f"Epoch {epoch_number} avg for {step} steps: train accuracy: {rolling_accuracy}, train loss: {rolling_loss}")
                rolling_loss = 0
                rolling_accuracy = 0

            d = {"adv-train/accuracy": accuracy, 
                "adv-train/loss": loss, 
                "adv-train/epoch": epoch_number
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

    def calc_loss_and_accuracy(self, logits, mask_label, input_ids):
        # CE loss
        mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        loss_func = CrossEntropyLoss()
        batch_size, seq_len, vocab_size = logits.shape
        mask_logits = logits[mask_indices]
        loss = loss_func(mask_logits.view(-1, vocab_size), mask_label.view(-1))
        loss = torch.mean(loss) # mean across batch

        accuracy = self.calc_accuracy(logits, mask_label, input_ids)
                    
        return loss, accuracy

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

        masked_sentence = example.masked_sentence
        if utils.MASK not in masked_sentence:
            continue

        input_ids = tokenizer.encode(masked_sentence,
                                        pad_to_max_length=True,
                                        max_length=args.max_seq_length,
                                        add_special_tokens=True)


        input_ids_list.append(input_ids)
        label_id_list.append(label_id)

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
    args = parse_adv_args()
    if "JOB_NAME" in os.environ:
        wandb_name = os.environ["JOB_NAME"]
    else:
        wandb_name = 'vscode'

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