epoch="5"
batch_size="1"
run_name="java_multi-lmhead-gpt2-with-exit"
exp_dir=CHANGE_HERE_TO_YOUR_EXPERIMENT_DIR
data_path="./datasets/java_multi-lmhead-gpt2_labeled"
cache_path="./cached/gpt2"
language="java"
model="multi-lmhead-gpt2-with-exit"
text_length="128"
checkpoint_path=CHANGE_HERE_TO_CHECKPOINT_PATH_OF_TRAINED_MODEL_IN_STEP_THREE
python eval.py --run_name ${run_name} --data_path ${data_path} --batch_size 1 --epoch ${epoch} --cache_path ${cache_path} --text_length ${text_length} --language ${language} --checkpoint_path ${checkpoint_path} --model ${model} --output_path ${exp_dir}