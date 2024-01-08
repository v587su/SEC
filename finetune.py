from model import CodeGenMultiLMHeadModelWithExit, GPT2MultiLMHeadModelWithExit 
import os
from model.gpt2_dataset import GPT2Dataset
from transformers import GPT2TokenizerFast, TrainingArguments, Trainer, GPT2LMHeadModel
import torch
from utils import arg_parser

def get_latest_checkpoint(dir_path):
    steps = []
    for file in os.listdir(dir_path):
        if file.startswith('checkpoint'):
            steps.append(int(file.split('-')[-1]))
    if len(steps) > 0:
        return f'checkpoint-{max(steps)}'
    else:
        return None


if __name__ == '__main__':
    args = arg_parser()
    print(args)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
    if args.model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(args.cache_path)
    elif args.model == 'multi-lmhead-gpt2':
        model = GPT2MultiLMHeadModelWithExit.from_pretrained(args.cache_path,mode='train_lm')
    elif args.model == 'multi-lmhead-gpt2-with-exit':
        model = GPT2MultiLMHeadModelWithExit.from_pretrained(args.checkpoint_path, mode='train_classifier')
    elif args.model == 'multi-lmhead-codegen':
        model = CodeGenMultiLMHeadModelWithExit.from_pretrained(args.cache_path, mode='train_lm')
    elif args.model == 'multi-lmhead-codegen-with-exit':
        model = CodeGenMultiLMHeadModelWithExit.from_pretrained(args.checkpoint_path, mode='train_classifier')
    else:
        raise NotImplementedError
    print(args)
    # check cuda status
    print('cuda status:', torch.cuda.is_available())
    model.to(args.device)
    dataset = GPT2Dataset(args.data_path, tokenizer, is_dev=args.is_dev, max_pos_length=args.text_length, language=args.language, mode='finetune' if args.model.endswith('exit') else 'pretrain')

    training_args = TrainingArguments(
        output_dir=args.output_path,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        save_strategy='epoch',
        save_total_limit=1,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.dataset["train"],
        eval_dataset=dataset.dataset["test"],
    )

    if args.load_from_checkpoint and os.path.exists(args.output_path) and get_latest_checkpoint(args.output_path):
        trainer.train(os.path.join(args.output_path, get_latest_checkpoint(args.output_path)))
    else:
        trainer.train()
