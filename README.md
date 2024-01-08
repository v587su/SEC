# Source code of the paper "Sizing up the Situation: Towards a Wise Code Completion Model"

### Datasets
Download CSN and COFIC into the `datasets` folder. The datasets are available at [CSN(python)](https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip) and [COFIC](https://drive.google.com/file/d/1Ai0WMYrIGQQLqBC180mzUVDSbpkgO6uD/view)

### LCMs
#### GPT-2
Download the pre-trained GPT-2 model from [here](https://huggingface.co/gpt2/tree/main) and put it into the `cached` folder.
The model needs to be finetuned on the dataset before using it for code completion. 
The command for finetuning GPT-2 on CSN(Python) is as follows:
```
python finetune.py --batch_size 8 --run_name python_gpt2 --epoch 10 --text_length 256 --cache_path ./cached/gpt2 --data_path datasets/python/final/jsonl --language python --output_path ./cached
```

The command for finetuning GPT-2 on COFIC(Java) is as follows:
```
python train_gpt2.py --batch_size 8 --run_name java_gpt2 --epoch 10  --text_length 256 --cache_path ./cached/gpt2  --data_path ./datasets/java/COFIC.jsonl --language java --output_path ./cached
```


#### CodeGen
Download the pre-trained CodeGen model from [here](https://huggingface.co/Salesforce/codegen-350M-mono/tree/main) and put it into the `cached` folder.
The model doesn't need further finetuning.

### Run
We have prepared the scripts for running the experiments in the paper. The scripts are in the `cmds` folder.
Please follow the steps and the instructions in the scripts to run the experiments.
