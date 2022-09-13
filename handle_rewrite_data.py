import enum
import os
import json
from selectors import EpollSelector
from utils import LamaExample, MASK, LamaExampleEncoder
from evaluation import filter_to_bert_success_only, predict_mask_token, evaluate
import argparse
import utils
from transformers import BertForMaskedLM, BertTokenizer
from faiss import IndexFlatL2
import torch
from tqdm import tqdm

def get_trex_data(lama_data_path, is_lpqa=False):
    lama_examples = {}
    relations, data_path_pre, data_path_post = get_trex_parameters(lama_data_path, is_lpqa)
    for relation in relations:
        file_path = data_path_pre+relation["relation"]+data_path_post
        if not os.path.exists(file_path):
            print(f"skipping {file_path}")
            continue

        if is_lpqa:
            relation_lama_examples = read_trex_lpaqa_relation(file_path, relation["label"], relation["template"])
        else:
            relation_lama_examples = read_trex_relation(file_path, relation["label"], relation["template"])

        lama_examples[relation["label"]] = (relation_lama_examples, relation["template"])
    return lama_examples

def get_trex_parameters(lama_data_path, is_lpqa):
    data_path_pre = lama_data_path
    folder_name = "TREx" if not is_lpqa else "TREx_train"
    relations = load_file("{}/relations.jsonl".format(data_path_pre))
    data_path_post = ".jsonl"
    data_path_pre = f"{data_path_pre}/{folder_name}/"

    return relations, data_path_pre, data_path_post

def read_trex_lpaqa_relation(filename, relation, masked_template):
    data = load_file(filename)

    lama_examples = []
    for sample in data:
        id = sample["sub_uri"]
        label = sample["obj_label"]
        sub_label = sample["sub_label"]
        new_masked_sentence = masked_template.replace("[X]", sub_label).replace("[Y]", MASK)
        masked_orig_sentence = new_masked_sentence
        lama_examples.append(LamaExample(id,
                                         "T-Rex",
                                         relation,
                                         masked_template,
                                         masked_orig_sentence,
                                         new_masked_sentence,
                                         sub_label,
                                         label.lower()))
    return lama_examples

def read_trex_relation(filename, relation, masked_template):
    data = load_file(filename)

    lama_examples = []
    for sample in data:
        id = sample["uuid"]
        label = sample["obj_label"]
        sub_label = sample["sub_label"]

        masked_orig_sentence = sample["evidences"][0]["masked_sentence"]
        new_masked_sentence = masked_template.replace("[X]", sub_label).replace("[Y]", MASK)
        lama_examples.append(LamaExample(id,
                                         "T-Rex",
                                         relation,
                                         masked_template,
                                         masked_orig_sentence,
                                         new_masked_sentence,
                                         sub_label,
                                         label.lower()))
    return lama_examples

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    return data


def save_all_lama_data():
    lama_examples = get_trex_data("/home/gamir/liat/data/lama")

    output_file_path = "/home/gamir/liat/UnberteseNew/data/all_lama_trex.json"
    json.dump(lama_examples, open(output_file_path, 'w'), indent=4, cls=LamaExampleEncoder)

def save_all_lpqa_data():
    lama_examples = get_trex_data("/home/gamir/liat/data/lpaqa", is_lpqa=True)

    output_file_path = "/home/gamir/liat/UnberteseNew/data/all_lpqa_trex.json"
    json.dump(lama_examples, open(output_file_path, 'w'), indent=4, cls=LamaExampleEncoder)

def save_correct_examples_only(args):
    lama_data = json.load(open(args.data_path, 'r'))
    device, n_gpu = utils.get_device()
    model = BertForMaskedLM.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir)
    filtered_examples = filter_to_bert_success_only(lama_data, model, tokenizer)
    json.dump(filtered_examples, open(f"args.output_dir/correct_examples", 'w'), indent=4, cls=LamaExampleEncoder)

def rewrite_valid_examples(examples, model, tokenizer, index, args, device, mlm_model=None):
    filtered_examples = []
    rewrite_success_counter = 0
    rewrite_counter = 0
    counter = 0
    invalid_counter = 0
    failed_counter = 0
    for i in tqdm(range(len(examples))):
        example = examples[i]
        label_id = tokenizer.encode(example['label'], add_special_tokens=False)

        if len(label_id) > 1:
            continue

        counter = counter + 1
        masked_sentence = example['masked_sentence']

        input_ids = torch.tensor(tokenizer.encode(masked_sentence,
                                        pad_to_max_length=True,
                                        max_length=args.max_seq_length,
                                        add_special_tokens=True), device=device).unsqueeze(0)
                                
        original_sentence=example['masked_sentence']
        with torch.no_grad():
            outputs = model.bert(input_ids=input_ids)

        last_hidden_states = outputs.last_hidden_state
        rewrite_embeds = last_hidden_states.detach().cpu().numpy()
        D, I = index.search(rewrite_embeds.reshape(-1, rewrite_embeds.shape[2]), 1)
        rewrited_sequence = I.reshape(rewrite_embeds.shape[0], -1)[0]
        rewrited_sequence = utils.bert_decode_clean(rewrited_sequence, tokenizer, filter_unused=True)
        original_sentence = original_sentence.lower().replace("[mask]", "[MASK]")
        rewrited_sequence = rewrited_sequence.replace(".", " .")
        example['masked_sentence'] = rewrited_sequence
        example['before_rewrite'] = original_sentence

        if rewrited_sequence != original_sentence:
            rewrite_counter = rewrite_counter + 1

            if mlm_model is not None:
                original_tokens_predictions = predict_mask_token(mlm_model, tokenizer, original_sentence)

                try:
                    rewrited_tokens_predictions = predict_mask_token(mlm_model, tokenizer, rewrited_sequence)
                except:
                    print(f"\nEvalution error for sentence: {rewrited_sequence}")
                    invalid_counter += 1
                    filtered_examples.append(example)
                    continue

                original_success = example['label'] in original_tokens_predictions
                rewrited_success = example['label'] in rewrited_tokens_predictions
                example['before_rewrite_success'] = original_success
                example['after_rewrite_success'] = rewrited_success

                if rewrited_success and not original_success:
                    print("-"*40)
                    print(f"Success!")
                    print(f"\nBefore: \"{original_sentence}\"")
                    print(f"After:  \"{rewrited_sequence}\"")
                    print("-"*40)
                    rewrite_success_counter = rewrite_success_counter + 1
                elif original_success and not rewrited_success:
                    print("-"*40)
                    print(f"Faliure!")
                    print(f"\nBefore: \"{original_sentence}\"")
                    print(f"After:  \"{rewrited_sequence}\"")
                    print("-"*40)
                    failed_counter += 1
                    filtered_examples.append(example)



    return filtered_examples, counter, rewrite_counter, rewrite_success_counter, invalid_counter, failed_counter

def unify_rewrited_files(args):
    files = [args.data_path, f"{args.output_dir}/rewrited_results"]
    tokenizer = BertTokenizer.from_pretrained(args.mlm_bert_model_name)
    result = {}
    for i, file in enumerate(files):
        lama_data = json.load(open(file, 'r'))
        data_relations = list(lama_data.keys())

        for r in tqdm(range(len(data_relations))):
            relation = data_relations[r]
            examples, relation_template = lama_data[relation]
            examples_to_add = []
            if relation in result:
                examples_to_add = result[relation][0]

            for example in examples:
                label_id = tokenizer.encode(example['label'], add_special_tokens=False)

                if len(label_id) > 1:
                    continue

                examples_to_add.append(example)

            result[relation] = (examples_to_add, relation_template)

    json.dump(result, open(f"{args.output_dir}/adv_training_data", 'w'), indent=4, cls=LamaExampleEncoder)


def save_bertese_rewrites(args):
    device, n_gpu = utils.get_device()
    lama_data = json.load(open(args.data_path, 'r'))

    model = BertForMaskedLM.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir).to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir)
    mlm_model = BertForMaskedLM.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir).to(device).eval()

    vocab_size = model.config.vocab_size
    vocab_indices = torch.tensor(list(range(vocab_size)), device=device)
    vocab_embeddings = model.bert.embeddings.word_embeddings(vocab_indices)
    index = IndexFlatL2(vocab_embeddings.shape[-1])
    index.add(vocab_embeddings.detach().cpu().numpy())

    saved_state = utils.load_states_from_checkpoint(args.rewriter_model_path)
    utils.load_saved_state(model, saved_state)

    results = {}
    data_relations = list(lama_data.keys())
    all_relations_counter = 0 
    all_relations_rewrite_counter = 0
    all_rewrite_success_counter = 0
    all_invalid_counter = 0
    all_failed_counter = 0

    for r in tqdm(range(len(data_relations))):
        relation = data_relations[r]
        examples, relation_template = lama_data[relation]

        filtered_examples, counter, rewrite_counter, rewrite_success_counter, invalid_counter, failed_counter = rewrite_valid_examples(examples, model, tokenizer, index, args, device, mlm_model)
        results[relation] = (filtered_examples, relation_template)
        all_relations_counter += counter
        all_relations_rewrite_counter += rewrite_counter
        all_rewrite_success_counter += rewrite_success_counter
        all_invalid_counter += invalid_counter
        all_failed_counter += failed_counter

    json.dump(results, open(f"{args.output_dir}/rewrited_results", 'w'), indent=4, cls=LamaExampleEncoder)
    print("*" *40)
    print(f"Total number of valid examples: {all_relations_counter}")
    print(f"Total number of rewrited examples: {all_relations_rewrite_counter} ~ {all_relations_rewrite_counter/all_relations_counter} of the data")
    print(f"Total number of rewrited changed to success examples: {all_rewrite_success_counter} ~ {all_rewrite_success_counter/all_relations_rewrite_counter} of the rwrited data")
    print(f"Total number of rewrited changed to failed examples: {all_failed_counter} ~ {all_failed_counter/all_relations_rewrite_counter} of the rwrited data")
    print(f"Total number of examples changed to invalid form: {all_invalid_counter} ~ {all_invalid_counter/all_relations_rewrite_counter} of the rwrited data")

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default="/home/gamir/liat/UnberteseNew/data/all_lama_trex.json",
                    type=str,
                    required=False)
parser.add_argument("--cache_dir",
                    default="/home/gamir/liat/Runs/cache_dir",
                    type=str,
                    required=False)
parser.add_argument("--rewriter_model_path",
                    default="/home/gamir/liat/Runs/unbertese220/checkpoints/unbertese.9",
                    type=str,
                    required=False)
parser.add_argument("--mlm_bert_model_name",
                    default="bert-base-uncased",
                    type=str,
                    required=False,
                    help="Bert pre-trained model selected in the list: lower_bert-base-uncased, "
                            "bert-large-uncased, lower_bert-base-cased, lower_bert-base-multilingual, lower_bert-base-chinese")
parser.add_argument("--output_dir",
                    default="/specific/netapp5_2/gamir/liat/data",
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length",
                    default=20,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
parser.add_argument("--evaluate",
                    default=False,
                    action='store_true')
parser.add_argument("--adv_ft_bert",
                    default="/home/gamir/liat/adv_Runs/unbertese_adv_unified/checkpoints/adv_unbertese.4",
                    type=str,
                    required=False)
args = parser.parse_args()

if not args.evaluate:
    save_bertese_rewrites(args)
    unify_rewrited_files(args)
else:
    evaluate(args)
#save_correct_examples_only('/home/gamir/liat/UnberteseNew/data/lama_correct_results.json', '/home/gamir/liat/UnberteseNew/data/all_lama_trex.json')