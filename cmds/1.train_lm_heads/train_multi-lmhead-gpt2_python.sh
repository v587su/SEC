epoch="10"
batch_size="12"
run_name="python_multi-lmhead-gpt2"
exp_dir=CHANGE_HERE_TO_YOUR_EXPERIMENT_DIR
data_path="./datasets/python/final/jsonl"
cache_path="./cached/python-gpt2"
language="python"
model="multi-lmhead-gpt2"
text_length="256"
python finetune.py --run_name ${run_name} --data_path ${data_path} --batch_size ${batch_size} --epoch ${epoch} --cache_path ${cache_path} --output_path ${exp_dir}/models/${run_name} --text_length ${text_length} --model ${model} --language ${language}