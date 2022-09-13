import nltk
import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
import logging
import os
from json import *
from json.decoder import  WHITESPACE
import collections
from torch.serialization import default_restore_location
import json 
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
import wandb
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

MASK = "[MASK]"
CLS = "[CLS]"
SEP = "[SEP]"
MASK_DUMMY = "mm"
CLS_DUMMY = "cc"
SEP_DUMMY = "ss"

PED_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
SPECIAL_IDS = [PED_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID]

UNUSED_IDS = [i for i in np.arange(998) if i not in SPECIAL_IDS]

class LamaExample:
    def __init__(self, uuid, source, relation, masked_template, snippet, masked_sentence,
                sub_label, label, valid_for_train=True):
        self.uuid = uuid
        self.source = source
        self.relation = relation
        self.masked_template = masked_template
        self.snippet = snippet
        self.masked_sentence = masked_sentence
        self.sub_label = sub_label
        self.label = label
        self.valid_for_train = valid_for_train
        self.unmasked_sentence = masked_sentence.replace("[MASK]", self.label)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [f"uuid: {self.uuid}\n",
             f"source: {self.source}\n",
             f"relation: {self.relation}\n",
             f"masked_template: {self.masked_template}\n",
             f"snippet: {self.snippet}\n",
             f"masked_sentence: {self.masked_sentence}\n",
             f"sub_label: {self.sub_label}\n",
             f"label: {self.label}\n",
             f"unmasked_sentence: {self.unmasked_sentence}\n",
             f"valid_for_train: {self.valid_for_train}\n",
             f"unmasked_sentence: {self.unmasked_sentence}"
             ]
        return " ".join(l)

    def to_dict(self):
        out = {"uuid": self.uuid,
               "source": self.source,
               "relation": self.relation,
               "masked_template": self.masked_template,
               "masked_sentence": self.masked_sentence,
               "snippet": self.snippet,
               "sub_label": self.sub_label,
               "label": self.label,
               "valid_for_train": self.valid_for_train,
               "unmasked_sentence": self.unmasked_sentence,}
        return out

    @classmethod
    def from_dict(cls, e_dict):
        return cls(e_dict["uuid"], e_dict["source"], e_dict["relation"], e_dict["masked_template"],
                           e_dict["snippet"], e_dict["masked_sentence"], e_dict["sub_label"],
                           e_dict["label"], e_dict["valid_for_train"])


# subclass JSONEncoder
class LamaExampleEncoder(JSONEncoder):
    def default(self, o):
        return o.to_dict()


class LamaExampleDecoder(JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        return super.decode(self, s, _w=_w)
        
def _remove_pads(value):
    value_end_char = min(value.index("[PAD]") if "[PAD]" in value else len(value), value.index("nivorousnivorous") if "nivorousnivorous" in value else len(value))
    value_no_pad = value[:value_end_char]
    return value_no_pad

def _log_sample(y, y_pred, x=None):
    if random.randint(0, 100) % 10 == 0:
        if x is not None:
            logger.info("\nX: {}\nY: {}\nPRED_Y: {}\n".format(x, y, y_pred))
        else:
            logger.info("\nY: {}\nPRED_Y: {}\n".format(y, y_pred))

def bleu_accuracy(preds, out_labels, do_logs=False, log_writer=None, input_tokens=None):
    n = len(preds)
    total_bleu = 0
    for i in range(n):
        y_label = _remove_pads(out_labels[i])
        y_pred = _remove_pads(preds[i])
        x = None
        if input_tokens is not None and len(input_tokens) > i:
            x = _remove_pads(input_tokens[i])
        if do_logs and log_writer is not None:
            _log_sample(log_writer,  y_label, y_pred, x)
        total_bleu += nltk.translate.bleu_score.sentence_bleu([y_pred], y_label)

    return total_bleu/n

def single_token_simple_accuracy(preds, out_labels):
    n = len(preds)
    correct = 0
    for i in range(n):
        if np.array_equal(out_labels[i].lower(), preds[i].lower()):
            correct += 1
    return correct/n

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    return device, n_gpu

def get_model_obj(model: torch.nn.Module):
    return model.module if hasattr(model, "module") else model

def to_examples_list(data):
    all_examples = []
    for relation in data:
        examples, relation_template = data[relation]
        if len(examples) > 0 and not isinstance(examples[0], LamaExample):
            lama_examples = []
            for e in examples:
                lama_examples.append(LamaExample.from_dict(e))
            examples = lama_examples
        all_examples += examples
    return all_examples

def compte_all_dist(a,b):
    """
    a_norm = a.norm(dim=2)[:, :, None]
    b_t = b.permute(0, 2, 1).contiguous()
    b_norm = b.norm(dim=2)[:, None]
    all_dist = torch.sqrt(torch.sum(a_norm + b_norm - 2.0 * torch.bmm(a, b_t)))
    """
    A = a
    B = b
    if len(a.shape) == 3:
        A = a.view(a.shape[0]*a.shape[1],a.shape[2])
    if len(b.shape) == 3:
        B = b.view(b.shape[0]*b.shape[1],b.shape[2])

    sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
    sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
    #
    #res = torch.sqrt(sqrA - 2 * torch.mm(A, B.t()) + sqrB)
    res = sqrA - 2 * torch.mm(A, B.t()) + sqrB
    if len(a.shape) == 3:
        res = res.view(a.shape[0],a.shape[1],res.shape[1])
    return res


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def print_args(args, output_dir=None):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")

    if output_dir is not None:
        with open(os.path.join(output_dir, "args.txt"), "w") as f:
            for key, val in sorted(vars(args).items()):
                keystr = "{}".format(key) + (" " * (30 - len(key)))
                f.write(f"{keystr}   {val}\n")

def bert_decode_clean(arr_ids, tokenizer, filter_unused=True):
    filtered_ids, valid = _filter_garbage(arr_ids, filter_unused)
    res = ""

    if not valid or len(filtered_ids) == 0:
        res += "---> Trimmed, Many Tokens are [unused]"

    filtered_ids = [i for i in filtered_ids if i != PED_ID]

    if filtered_ids[0] == CLS_ID:
        filtered_ids = filtered_ids[1:]

    if filtered_ids[-1] == SEP_ID:
        filtered_ids = filtered_ids[:-1]

    res = tokenizer.decode(filtered_ids)
    return res

def _filter_garbage(ids, filter_unused):
    if filter_unused:
        no_unused_ids = [i for i in ids if i not in UNUSED_IDS]
        if len(ids) - len(no_unused_ids) < 20: #if we have less than 20 unknown tokens we consider it to be valid
            return no_unused_ids, True #todo: might be better to trim at the first unknown
        else:
            return no_unused_ids, False
    else:
        return ids, True

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
    ],
)

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(
        model_file, map_location=lambda s, l: default_restore_location(s, "cpu")
    )
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)

def load_saved_state(model, saved_state: CheckpointState):
    model_to_load = get_model_obj(model)
    logger.info("Loading saved model state ...")
    model_to_load.load_state_dict(
        saved_state.model_dict, strict=False
    )  # set strict=False if you use extra projection

    # if load_only_model:
    #     logger.info("We don't load optimizer, scheduler and step details..")
    #     return

    # epoch = saved_state.epoch
    # offset = 0#saved_state.offset
    # if offset == 0:  # epoch has been completed
    #     epoch += 1
    # logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

    # self.start_epoch = epoch
    # self.start_batch = offset

    # if saved_state.optimizer_dict:
    #     logger.info("Loading saved optimizer state ...")
    #     self.optimizer.load_state_dict(saved_state.optimizer_dict)

    # if saved_state.scheduler_dict:
    #     self.scheduler_state = saved_state.scheduler_dict


def mean_pooling(token_embeddings, mask):
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    result = torch.sum(token_embeddings * input_mask_expanded, 1) 
    result = result / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return result

def load_data(args, device, tokenizer):
    logger.info("Loading data")
    test_data = json.load(open(args.test_data_path, 'r'))
    train_data = json.load(open(args.train_data_path, 'r'))

    train_examples = to_examples_list(train_data)
    test_examples = to_examples_list(test_data)

    train_dataloader, train_dataset = load_lama_examples(args, tokenizer, train_examples, is_dev=False)
    test_dataloader, test_dataset = load_lama_examples(args, tokenizer, test_examples, is_dev=True)
    number_of_train_examples = len(train_examples)

    return train_dataloader, test_dataloader, number_of_train_examples

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

        input_ids = tokenizer.encode(masked_sentence,
                                        pad_to_max_length=True,
                                        max_length=args.max_seq_length,
                                        add_special_tokens=True)
        if MASK_ID not in input_ids:
            continue

        label_id_list.append(label_id)
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