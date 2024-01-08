import argparse
import random
import os
import re


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_dev', action='store_true', help='development mode') 
    parser.add_argument('--load_from_checkpoint', action='store_true', help='development mode') 
    parser.add_argument('--save_cases', action='store_true', help='development mode') 
    parser.add_argument('--use_tcqe', action='store_true', help='development mode') 
    parser.add_argument('--run_name', type=str, help='development mode') 
    parser.add_argument('--data_path', type=str, default='./datasets', help='Path to the dataset') 
    parser.add_argument('--language', type=str, default='python', help='Path to the dataset') 
    parser.add_argument('--model', type=str, default='gpt2', help='Path to the dataset') 
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch Number')
    parser.add_argument('--text_length', type=int, default=128, help='Length of the text')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Length of the text')
    parser.add_argument('--cache_path', type=str, default="./cached/gpt2", help='Length of the text')
    parser.add_argument('--stop_threshold', type=str, default='[0,0.5,0.9,1]', help='Length of the text')
    parser.add_argument('--exit_threshold', type=str, default='[0,0.5,0.9,1]', help='Length of the text')
    parser.add_argument('--output_path', type=str, default="./cached/gpt2", help='Length of the text')
    parser.add_argument('--device', type=str, default="cuda:0", help='Length of the text')
    return parser.parse_args()


def write_annotate_file(prompts, answer, generated,tokenizer, output_path):
    assert len(prompts) == len(generated) 
    assert len(prompts) == len(answer) 
    os.makedirs(os.path.join(output_path),exist_ok=True)
    for i, (p,g,a) in enumerate(zip(prompts, generated, answer)):
        p_str = tokenizer.decode(p)
        g_str = tokenizer.decode(g)
        a_str = tokenizer.decode(a)
        
        code = wrap_code(p_str)
        code += '\n\n =======================\n\n'
        code += f'answer: |{a_str}|\n'
        code += f'completion: |{g_str}|'
        code += '\n\n =======================\n\n'
        with open(f'{output_path}/{i}.java','w') as f:
            f.write(code)


def wrap_code(code, with_bracket=False):
    code = re.sub(r'(?<=[\S])([ ]{4,})(?=[\S])', r'\n\1', code)
    code = re.sub(r'(?<=[\S])([\t]{1,})(?=[\S])', r'\n\1', code)
    pre_fix = f'public class HumanEval {{\n'
    post_fix = '\n}\n' if with_bracket else ''
    return pre_fix+code+post_fix