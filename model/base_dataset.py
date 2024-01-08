import pickle
import os
import json
import random
from datasets import load_from_disk, concatenate_datasets

class BaseDataset:
    def __init__(self, data_path, tokenizer, is_dev=False,max_pos_length=256, language='python'):

        self.dataset = None
        self.tokenizer = tokenizer
        self.is_dev = is_dev
        self.language = language
        self.random_seed = 42
        self.max_pos_length = max_pos_length
        if data_path.endswith('.pickle'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:100] if is_dev else self.data
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                lines = f.readlines()
                lines = lines[:100] if is_dev else lines
                self.data = [json.loads(line) for line in lines]
        elif 'final' in data_path:
            train_data = []
            test_data = []
            for file in os.listdir(os.path.join(data_path,'train')):
                if file.endswith('.jsonl'):
                    with open(os.path.join(data_path,'train', file), 'r') as f:
                        lines = f.readlines()
                    lines = lines[:100] if is_dev else lines
                    train_data += [json.loads(line) for line in lines]
                    if is_dev:
                        break
            for file in os.listdir(os.path.join(data_path,'test')):
                if file.endswith('.jsonl'):
                    with open(os.path.join(data_path, 'test',file), 'r') as f:
                        lines = f.readlines()
                    lines = lines[:100] if is_dev else lines
                    test_data += [json.loads(line) for line in lines]
            self.data = {'train': train_data, 'test': test_data}
        elif os.path.isdir(data_path):
            shards = [load_from_disk(os.path.join(data_path, f'train_{i}')) for i in range(1)] 
            train_set = concatenate_datasets(shards)
            self.dataset = {
                'train': train_set,
                'test': load_from_disk(os.path.join(data_path, 'test'))
            }
            if is_dev:
                self.dataset['train'] = self.dataset['train'].shard(num_shards=2000,index=0)
                self.dataset['test'] = self.dataset['test'].shard(num_shards=2000,index=0)
        else:
            print(data_path)
            raise ValueError
    
        
    def train_test_split(self):
        assert self.dataset
        self.dataset = self.dataset.train_test_split(test_size=0.1, seed=self.random_seed)
        print(f'Train number: {str(len(self.dataset["train"]))}')
        print(f'Test number: {str(len(self.dataset["test"]))}')


    def sequential(self, name, ops=[]):
        for op in ops:
            if isinstance(op, list):
                self.dataset[name] = self.dataset[name].map(lambda x: getattr(self, op[0])(x), batched=True, load_from_cache_file=False, remove_columns=op[1])
            else:
                self.dataset[name] = self.dataset[name].map(lambda x: getattr(self, op)(x), batched=True, load_from_cache_file=False)

