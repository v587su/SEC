from model import GPT2MultiLMHeadModelWithExit, CodeGenMultiLMHeadModelWithExit
import pandas as pd
import time
from collections import defaultdict
import os
import numpy as np
import datasets
from model.gpt2_dataset import GPT2Dataset
from transformers import GPT2TokenizerFast
import tqdm
import random
from utils import arg_parser, write_annotate_file
import torch
from torch.utils.data import DataLoader


def next_token_classify(model, data_loader):
    results = []
    for batch in tqdm.tqdm(data_loader,desc=f'{model.__class__.__name__} generating'):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        cls_logits = model(input_ids=input_ids, attention_mask=attention_mask,return_exit_logits=True).logits
        each_layer_results = []
        for layer_logits in cls_logits:
            layer_logits = torch.softmax(layer_logits, dim=-1).reshape((-1,layer_logits.shape[-1]))
            each_layer_results.append(layer_logits.detach().cpu().numpy().tolist())
        results.append(each_layer_results)
    return results

def multiple_token_generate(model, data_loader, tokenizer, max_new_tokens=10, exit_thresholds=[], stop_thresholds=[], est_model=None, save_cases=False):
    generated_tokens, used_layers, times = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    stopped_continue = defaultdict(list)
    stopped_count = defaultdict(int)
    est_scores = []
    est_times = []
    
    
    for batch in tqdm.tqdm(data_loader, desc=f'{model.__class__.__name__} generating'):
        input_ids = batch["input_ids"].to(model.device)
        for i, exit_thre in enumerate(exit_thresholds): 
            for j, stop_thre in enumerate(stop_thresholds):
                if save_cases and i != j:
                    continue
                if not save_cases and exit_thre != 100 and stop_thre != 100:
                    continue
                if stop_thre not in generated_tokens[exit_thre].keys():
                    generated_tokens[exit_thre][stop_thre] = []
                    used_layers[exit_thre][stop_thre] = []
                    times[exit_thre][stop_thre] = []
                
                if exit_thre == 100 and stop_thre == 100 and est_model is not None:
                    start_time = time.time()
                    est_logits = est_model(input_ids).logits
                    est_scores.append(est_logits[0][0].item())
                    est_times.append(time.time() - start_time)

                start_time = time.time()
                outputs = model.generate(
                    input_ids, max_new_tokens=max_new_tokens, exit_threshold=exit_thre, stop_threshold=stop_thre, use_exit=True, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,min_length=0
                )
                recorded_time = time.time() - start_time

                if stop_thre != 100:
                    if outputs.is_stopped:
                        stopped_count[stop_thre] += 1
                generated = outputs.sequences[0, input_ids.size()[-1]:].cpu().numpy().tolist()
                layers = outputs.exit_layers
                generated_tokens[exit_thre][stop_thre].append(generated)
                used_layers[exit_thre][stop_thre].append(layers)
                times[exit_thre][stop_thre].append(recorded_time)
        
    return generated_tokens, used_layers, stopped_continue, times, stopped_count, est_scores, est_times


def compute_single_token_accuracy(labels, predict_logits, target_labels=[0,1], thresholds=[]):
    results = []
    thre_values = {}
    for thr in thresholds:
        predict_labels = []
        for prediction in predict_logits:
            current_predict_labels = []
            for layer in prediction:
                current_layer_labels = []
                for token in layer:
                    if token[0] > thr and token[0] == max(token):
                        current_layer_labels.append(0)
                    elif token[1] > thr and token[1] == max(token):
                        current_layer_labels.append(1)
                    else:
                        current_layer_labels.append(2)
                current_predict_labels.append(current_layer_labels)
            predict_labels.append(current_predict_labels)
        thre_values[thr] = predict_labels

    for thr in thresholds:
        for target_label in target_labels:
            all_TP, all_FP, all_TN, all_FN = 0, 0, 0, 0
            for label, prediction in zip(labels, thre_values[thr]):
                for layer_id, (each_layer_l, each_layer_p) in enumerate(zip(label, prediction)):
                    TP, FP, TN, FN = 0, 0, 0, 0
                    for l,p in zip(each_layer_l, each_layer_p):
                        if p == target_label:
                            if l == target_label:
                                TP += 1
                                all_TP += 1
                            else:
                                FP += 1
                                all_FP += 1
                        else:
                            if l == target_label:
                                FN += 1
                                all_FN += 1
                            else:
                                TN += 1
                                all_TN += 1
                 
            all_precision = all_TP / (all_TP + all_FP + 1e-8)
            all_recall = all_TP / (all_TP + all_FN + 1e-8)
            all_F1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-8)
            results.append({'threshold':thr, 'label': target_label,"precision":all_precision,"recall":all_recall,"F1":all_F1,'TP':all_TP,'FP':all_FP,'TN':all_TN,'FN':all_FN,'TP+FP':all_TP+all_FP,'TP+FN':all_TP+all_FN})
    return results

def compute_multi_token_accuracy(ground_truth, generated, stopped_continue, used_layers, times, stopped_count, tokenizer):
    bleu = datasets.load_metric('./cached/bleu/bleu.py')
    rouge = datasets.load_metric('./cached/rouge/rouge.py')
    ground_truth = [[[str(i) for i in item]] for item in ground_truth] 

    results = []
    for exit_thre in generated.keys():
        for stop_thre in generated[exit_thre].keys():

            current = generated[exit_thre][stop_thre]
            used_layer = used_layers[exit_thre][stop_thre]
            used_time = times[exit_thre][stop_thre]
            current = [[str(i) for i in item if i != tokenizer.eos_token_id] for item in current]
            stop_count = stopped_count[stop_thre]
            bleu_score = bleu.compute(predictions=current, references=ground_truth)['bleu']
            rouge_score = rouge.compute(predictions=current, references=ground_truth)['rougeL'].mid.fmeasure
          
            average_length = sum([len(i) for i in current]) / len(current)
            average_layer = sum([sum(i)/(len(i)+1e-8) for i in used_layer]) / len(used_layer)
            average_time = sum(used_time) / len(used_time)
            results.append({"exit_threshold":exit_thre,"stop_threshold":stop_thre,"bleu":bleu_score,"rouge":rouge_score, "average_length":average_length,"average_layer":average_layer, "average_time": average_time, "stop_count": stop_count})
    return results


def sample_and_save(data_loader, generated, answers,save_settings, args, tokenizer, est_scores=None):
    prompts = [b['input_ids'][0].numpy().tolist() for b in data_loader]
    if est_scores is not None:
        for thre in save_settings:
            tmp_generated = [generated[i] for i,s in enumerate(est_scores) if s > thre]
            tmp_prompts = [prompts[i] for i,s in enumerate(est_scores) if s > thre]
            tmp_answers = [answers[i] for i,s in enumerate(est_scores) if s > thre]
            index = random.sample(list(range(len(prompts))), 379 if not args.is_dev else 1)
            tmp_prompts = [tmp_prompts[i] for i in index]
            tmp_answers = [tmp_answers[i] for i in index]
            tmp_generated  = [tmp_generated[e_thre][s_thre][i] for i in index]
            write_annotate_file(tmp_prompts, tmp_answers, tmp_generated, f'{args.output_path}/human/est_{thre}')
    else:
        for e_thre, s_thre in save_settings:
            index = random.sample(list(range(len(prompts))), 379 if not args.is_dev else 1)
            tmp_prompts = [prompts[i] for i in index]
            tmp_answers = [answers[i] for i in index]
            tmp_generated  = [generated[e_thre][s_thre][i] for i in index]
            write_annotate_file(tmp_prompts, tmp_answers, tmp_generated, tokenizer, f'{args.output_path}/human/{args.model}_{e_thre}_{s_thre}')


if __name__ == '__main__':
    args = arg_parser()
    random.seed(233)
    print(args)
    thresholds = [0.0, 0.5, 0.6, 0.7,0.8, 0.9, 0.95, 0.99,0.999] 
    est_settings = [0.269, 0.531]
    args.exit_thresholds = thresholds
    args.stop_thresholds = thresholds
    args.max_new_tokens = 10
    tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
    est_model = None
    if args.model == 'multi-lmhead-codegen-with-exit':
        model = CodeGenMultiLMHeadModelWithExit.from_pretrained(args.checkpoint_path)
        layers = list(range(1, 20, 3))
        rq3_thresholds = []
        if args.save_cases: 
            if args.language == 'java':
                args.exit_thresholds = [100]
                args.stop_thresholds = [100]
            elif args.language == 'python':
                args.exit_thresholds = [0.95,0.9,0.8,0.6]
                args.stop_thresholds = [0.8,0.7,0.6,0.5]

    elif args.model == 'multi-lmhead-gpt2-with-exit':
        model = GPT2MultiLMHeadModelWithExit.from_pretrained(args.checkpoint_path)
        layers = list(range(1, 12))
        if args.save_cases: 
            if args.language == 'java':
                args.exit_thresholds = [0.999,0.95,0.9]
                args.stop_thresholds = [0.8, 0.6, 0.0]
            elif args.language == 'python':
                args.exit_thresholds = [0.95,0.9, 0.8, 0.6]
                args.stop_thresholds = [0.9, 0.7, 0.6, 0.0]

    else:
        raise ValueError("model name not supported")
    


    model.to(args.device)
    model.eval()
    dataset = GPT2Dataset(args.data_path, tokenizer, is_dev=args.is_dev, max_pos_length=args.text_length, max_new_tokens=args.max_new_tokens,language=args.language, mode='eval')
    multi_token_test_data = dataset.dataset["multi_token_test"].with_format("torch")
    multi_loader = DataLoader(multi_token_test_data, batch_size=args.batch_size, shuffle=False)
    generated, layers, stopped_continue, times, stopped_count,est_scores, est_times = multiple_token_generate(model, multi_loader, tokenizer, max_new_tokens=args.max_new_tokens,exit_thresholds=args.exit_thresholds, stop_thresholds=args.stop_thresholds, save_cases=args.save_cases, est_model=est_model)
    if args.save_cases and args.language == 'java':
        sample_and_save(multi_loader, generated, dataset.dataset["multi_token_test"]["answers"], [(i,j) for i,j in zip(args.exit_thresholds, args.stop_thresholds)], args, tokenizer)
    multi_results = compute_multi_token_accuracy(dataset.dataset["multi_token_test"]["answers"], generated, stopped_continue, layers, times, stopped_count, tokenizer)
    for row in multi_results:
        print(row)
    results_pd = pd.DataFrame(multi_results)    
   
    results_pd.to_csv(f"./{args.output_path}/{args.language}_{args.model}_multi_token_results.csv" if not args.save_cases else f"./{args.output_path}/rq3_{args.language}_{args.model}_multi_token_results.csv", index=False)
    if not args.save_cases:
        single_token_test_data = dataset.dataset["single_token_test"].with_format("torch")
        single_loader = DataLoader(single_token_test_data, batch_size=args.batch_size, shuffle=False)
        cls_logits = next_token_classify(model, single_loader)
        answers = dataset.dataset["single_token_test"]["labels"]
        single_results = compute_single_token_accuracy(answers, cls_logits, target_labels=[0,1], thresholds=thresholds) 
        results_pd = pd.DataFrame(single_results)
        results_pd.to_csv(f"./{args.output_path}/{args.language}_{args.model}_single_token_results.csv", index=False)
    