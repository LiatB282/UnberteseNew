from tqdm import tqdm
import torch
from utils import MASK
import utils
import json
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np

def calc_accuracy(tokenizer, logits, label_ids, input_ids, average=False):
        mask_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_logits = logits[mask_indices].cpu().numpy()
        predictions = np.argmax(mask_logits, axis=1)
        n = len(predictions)
        accuracy = 0
        for i in range(n):
            if label_ids[i] == predictions[i]:
                accuracy += 1
        if average:
            accuracy = accuracy / n
        return accuracy

def evaluate_bert(lama_data, model, tokenizer):
    accuracy = 0.0
    data_relations = list(lama_data.keys())

    for r in tqdm(range(len(data_relations))):
        relation_accuracy = 0
        relation = data_relations[r]
        examples, relation_template = lama_data[relation]
        batch_size = 32
        for i in range(0, len(examples), batch_size):
            batch_start = i
            batch_end = min(batch_start + batch_size, len(examples))
            batch_examples = examples[batch_start:batch_end]
            batch_masked_sentences = [e['masked_sentence'] for e in batch_examples]
            batch_input = tokenizer(batch_masked_sentences,
                                        padding=True,
                                        add_special_tokens=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**batch_input).logits
            batch_labels = [e["label"] for e in batch_examples] 
            batch_label_ids = tokenizer(batch_labels, add_special_tokens=False, return_tensors="pt")['input_ids'].squeeze(1)

            relation_accuracy += calc_accuracy(tokenizer, logits, batch_label_ids, batch_input["input_ids"], average=False)
        relation_accuracy = relation_accuracy / len(examples)
        accuracy += relation_accuracy

    # Macro - average
    accuracy = accuracy / len(data_relations)

    return accuracy

def filter_to_bert_success_only(data, model, tokenizer):
    results = {}
    data_relations = list(data.keys())
    for r in tqdm(range(len(data_relations)), desc="Filter Relation Data"):
        relation = data_relations[r]
        filtered_examples = []
        examples, relation_template = data[relation]
        for i in range(len(examples)):
            example = examples[i]
            bert_pred = predict_mask_token(model=model, tokenizer=tokenizer, sentence=example['masked_sentence'])
            if example['label'] in bert_pred:
                filtered_examples.append(example)
        results[relation] = (filtered_examples, relation_template)
    return results

def predict_mask_token(model, tokenizer, sentence, top_k=10):
    if not (sentence).count(MASK) == 1:
        print("ERORR!" + sentence)
        raise ValueError("there should be exactly one MASK")
    input_ids = torch.tensor(tokenizer.encode(sentence), device=device).unsqueeze(0)
    attention_mask = input_ids != tokenizer.pad_token_id
    token_type_ids = torch.zeros_like(input_ids, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]    
    predictions = logits[0, mask_token_index]

    topk_predictions = torch.topk(predictions, 10)[1]
    predicted_tokens = tokenizer.convert_ids_to_tokens(topk_predictions[0])

    return predicted_tokens

def print_accuracy(lama_data, model, tokenizer):
    accuracy = evaluate_bert(lama_data, model, tokenizer)
    print('*' * 40)
    print(f"Accuracy:  {accuracy}")
    print('*' * 40)

def evaluate(args):
    lama_data = json.load(open(args.data_path, 'r'))
    model = BertForMaskedLM.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.mlm_bert_model_name, cache_dir=args.cache_dir)
    saved_state = utils.load_states_from_checkpoint(args.adv_ft_bert)
    utils.load_saved_state(model, saved_state)
    print_accuracy(lama_data, model, tokenizer)

device, n_gpu = utils.get_device()

