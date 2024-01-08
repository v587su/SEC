epoch="10"
batch_size="1"
run_name="java_multi-lmhead-codegen"
exp_dir=CHANGE_HERE_TO_YOUR_EXPERIMENT_DIR
data_path="./datasets/java/COFIC.jsonl"
cache_path="./cached/codegen-350M-multi"
language="java"
model="multi-lmhead-codegen"
text_length="128"
checkpoint_path=CHANGE_HERE_TO_CHECKPOINT_PATH_OF_TRAINED_MODEL_IN_STEP_ONE
python generate_dataset.py --run_name ${run_name} --data_path ${data_path} --batch_size 1 --epoch ${epoch} --cache_path ${cache_path} --output_path ${exp_dir}/models/${run_name} --text_length ${text_length} --model ${model} --language ${language} --checkpoint_path ${checkpoint_path}