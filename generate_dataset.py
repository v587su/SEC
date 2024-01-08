from utils import arg_parser
import numpy as np
from collections import defaultdict
from transformers import GPT2TokenizerFast
from model.gpt2_dataset import GPT2Dataset
from model import GPT2MultiLMHeadModelWithExit, CodeGenMultiLMHeadModelWithExit
from torch.utils.data import DataLoader
import torch
import tqdm
from PIL import Image, ImageDraw
from colour import Color

def teacher_forcing_generate(model, data_loader):
    results = defaultdict(list)
    for batch in tqdm.tqdm(data_loader,desc=f'{model.__class__.__name__} generating'):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["input_ids"].to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_all_layers=True)
        num_layers = len(outputs.logits)
        for i in range(num_layers):
            logits = outputs.logits[i]
            generated_token_ids = torch.argmax(logits, dim=-1)
            results[i].append(generated_token_ids.cpu().numpy().reshape(-1).tolist()) 
    return results


def compute_score(all_layer_generated, ground_truth_list):
    first_correct_layers = []
    labels = []

    for i in range(len(ground_truth_list)):
        ground_truth = ground_truth_list[i][1:]
        first_correct_layer = [0] * len(ground_truth)
        for layer_id in range(len(all_layer_generated.keys())):
            layer_generated = all_layer_generated[layer_id][i]
            for j, (gen_token, ans_token) in enumerate(zip(layer_generated, ground_truth)):
                if first_correct_layer[j] != 0:
                    continue
                if gen_token == ans_token:
                    first_correct_layer[j] = layer_id + 1
        first_correct_layers.append(first_correct_layer)
    
    for first_correct_layer in first_correct_layers:
        all_layer_labels = [[] for _ in range(len(all_layer_generated))]
        for layer_id in first_correct_layer:
            for i in range(len(all_layer_labels)):
                if layer_id == 0:
                    all_layer_labels[i].append(0)
                elif layer_id <= i + 1:
                    all_layer_labels[i].append(1)
                else:
                    all_layer_labels[i].append(2)
                # 0 is hard, 1 is easy, 2 is medium
        labels.append(all_layer_labels)
    # labels: [sample_num, layer_num, token_num] 
    return labels, first_correct_layers

 
def count_label_frequency(labels):
    counter = defaultdict(int)
    for label_seq in labels:
        for label in label_seq:
            counter[label] += 1
    return counter


if __name__ == '__main__':
    args = arg_parser()
    print(args)
    
    tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
    dataset = GPT2Dataset(args.data_path, tokenizer, is_dev=args.is_dev, max_pos_length=args.text_length, language=args.language, mode='generate')
    test_data_loader = dataset.dataset['test'].with_format('torch')
    test_data_loader = DataLoader(test_data_loader, batch_size=args.batch_size, shuffle=False)
    if 'gpt2' in args.model:
        model = GPT2MultiLMHeadModelWithExit.from_pretrained(args.checkpoint_path).to(args.device)
    elif 'codegen' in args.model:
        model = CodeGenMultiLMHeadModelWithExit.from_pretrained(args.checkpoint_path).to(args.device)
    else:
        raise ValueError('model name should be gpt2 or codegen')
    model.eval()
    for shard in [0,1]:
        current_shard = dataset.dataset['train'].shard(num_shards=2,index=shard)
        train_data_loader = current_shard.with_format('torch')
        print(len(train_data_loader))
        train_data_loader = DataLoader(train_data_loader, batch_size=args.batch_size, shuffle=False)
        train_token_generated = teacher_forcing_generate(model, train_data_loader)
        train_labels,train_first_correct = compute_score(train_token_generated, current_shard['input_ids'])
        print(count_label_frequency(train_first_correct))
        train_dataset = current_shard.add_column('labels', train_labels)
        train_dataset.save_to_disk(f'./datasets/{args.language}_{args.model}_labeled/train_{shard}' if not args.is_dev else f'./datasets/{args.language}_{args.model}_labeled_dev/train_{shard}')


    test_token_generated = teacher_forcing_generate(model, test_data_loader)
    test_labels,test_first_correct = compute_score(test_token_generated, dataset.dataset['test']['input_ids'])
    test_dataset = dataset.dataset['test'].add_column('labels', test_labels)
    test_dataset.save_to_disk(f'./datasets/{args.language}_{args.model}_labeled/test' if not args.is_dev else f'./datasets/{args.language}_{args.model}_labeled_dev/test')



