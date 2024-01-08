epoch="10"
batch_size="1"
run_name="java_multi-lmhead-gpt2-with-exit"
exp_dir=CHANGE_HERE_TO_YOUR_EXPERIMENT_DIR
data_path="./datasets/java_multi-lmhead-gpt2_labeled"
cache_path="./cached/gpt2"
language="java"
model="multi-lmhead-gpt2-with-exit"
text_length="256"
checkpoint_path=CHANGE_HERE_TO_CHECKPOINT_PATH_OF_TRAINED_MODEL_IN_STEP_ONE
python finetune.py --run_name ${run_name} --data_path ${data_path} --batch_size ${batch_size} --epoch ${epoch} --cache_path ${cache_path} --output_path ${exp_dir}/models/${run_name} --text_length ${text_length} --model ${model} --language ${language} --checkpoint_path ${checkpoint_path}