from multiprocessing.sharedctypes import Value
from .base_dataset import BaseDataset
import random
from datasets import Dataset
random.seed(42)

class GPT2Dataset(BaseDataset):
    def __init__(self, data_path, tokenizer, is_dev=False, max_pos_length=256, max_new_tokens=10, language='python', mode='pretrain'):
        super().__init__(data_path, tokenizer, is_dev=is_dev, max_pos_length=max_pos_length, language=language)
        self.max_new_tokens = max_new_tokens
        random.seed(self.random_seed)
        if self.dataset is None:
            if self.language == 'java':
                code =  [d['code'].strip() for d in self.data if len(d['code'].strip()) > 0]
                remove_dumplicated = list(set(code))
                self.dataset = Dataset.from_dict({
                    "code": remove_dumplicated,
                })
                self.train_test_split()
            elif self.language == 'python':
                self.dataset = {
                    'train': Dataset.from_dict({
                        "code": [d['code'].strip() for d in self.data['train'] if len(d['code'].strip()) > 0],
                    }),
                    'test': Dataset.from_dict({
                        "code": [d['code'].strip() for d in self.data['test'] if len(d['code'].strip()) > 0],
                    }),
                }
            else:
                raise ValueError('Language not supported')

        if mode == 'pretrain':
            self.sequential('train', ['add_eos',['tokenize_and_concate', 'code']])
            self.sequential('test', ['add_eos',['tokenize_and_concate', 'code']])
        elif mode == 'generate':
            self.sequential('train', [['tokenize', 'code']])
            self.sequential('test', [['tokenize', 'code']])
        elif mode == 'finetune':
            self.sequential('train', ['align_labels'])
            self.sequential('test', ['align_labels'])
        elif mode == 'eval':
            self.dataset['multi_token_test'] = self.dataset['test']
            self.sequential('multi_token_test', ['split'])
            self.dataset['single_token_test'] = self.dataset['test']
        else:
            raise ValueError('Mode not supported')
        
    def tokenize(self, examples):
        return self.tokenizer(examples['code'], max_length=self.max_pos_length, truncation=True)
    
    def align_labels(self, examples):
        examples['label'] = examples['labels']
        new_input_ids = []
        new_attention_masks = []
        for i in examples['input_ids']:
            new_input_ids.append(i[:-1])
        for i in examples['attention_mask']:
            new_attention_masks.append(i[:-1])
        examples['input_ids'] = new_input_ids
        examples['attention_mask'] = new_attention_masks
        return examples

    def split(self, examples):
        new_input_ids = []
        answers = []
        for input_ids in examples['input_ids']:
            split_point = random.randint(min(len(input_ids)-self.max_new_tokens, 50), len(input_ids)-1)
            former_part = input_ids[:split_point]
            latter_part = input_ids[split_point:split_point+self.max_new_tokens]
            new_input_ids.append(former_part)
            answers.append(latter_part)
        return {'input_ids':new_input_ids, 'answers':answers}


    def tokenize_and_concate(self, examples):
        tokenized_example = self.tokenizer(examples['code'])
        concatenated_examples = {}
        for k in tokenized_example.keys():
            concatenated_examples[k] = sum(tokenized_example[k], [])
          
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        result = {k:[] for k in concatenated_examples.keys()}
        for k,t in concatenated_examples.items():
            for i in range(0, total_length, self.max_pos_length):
                if i+self.max_pos_length < total_length:
                    result[k].append(t[i:i+self.max_pos_length])
        result["labels"] = result["input_ids"].copy()
        return result

    def add_eos(self, examples):
        new_code = []
        for c in examples['code']:
            new_code.append(c + self.tokenizer.eos_token)
        examples['code'] = new_code
        return examples 