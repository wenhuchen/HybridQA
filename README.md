# HybridQA
This repository contains the dataset and code for paper "HybridQA: A Dataset of Multi-Hop Question Answeringover Tabular and Textual Data"


# Running Stage1:
Running training command for stage1 as follows:

```
CUDA_VISIBLE_DEVICES=0 python train_stage12.py --do_lower_case --do_train --train_file ../stage1_training_data.json.gz --predict_file ../stage1_dev_data.json.gz --learning_rate 2e-6 --option stage1 --num_train_epochs 3.0
```

Running training command for stage2 as follows:
```
CUDA_VISIBLE_DEVICES=0 python train_stage12.py --do_lower_case --do_train --train_file ../stage2_training_data.json.gz --predict_file ../stage2_dev_data.json.gz --learning_rate 5e-6 --option stage2 --num_train_epochs 3.0
```

Running training command for stage3 as follows:
```
CUDA_VISIBLE_DEVICES=0 python train_stage3.py --do_train  --do_lower_case   --train_file ../stage3_training_data.json.gz   --predict_file ../stage3_dev_data.json.gz   --per_gpu_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 4.0   --max_seq_length 384   --doc_stride 128  --threads 8
```

Model Selction command for stage1 and stage2 as follows:
```
CUDA_VISIBLE_DEVICES=0 python train_stage12.py --do_lower_case --do_eval --option stage1 --output_dir stage1/2020_04_01_20_03_35/ --predict_file ../stage1_dev_data.json.gz
```

Evaluating command for stage1 and stage2 as follows:
```
CUDA_VISIBLE_DEVICES=0 python train_stage12.py --stage1_model stage1/2020_03_29_14_32_54/checkpoint/ --stage2_model stage2/2020_03_29_14_33_28/checkpoint/ --do_lower_case --predict_file ../dev_input.json --do_eval --option stage12
```

Evaluating command for stage3 as follows:
```
CUDA_VISIBLE_DEVICES=0 python train_stage3.py --model_name_or_path stage3/2020_03_27_23_44_59/checkpoint/ --do_stage3   --do_lower_case  --predict_file predictions.intermediate.json --per_gpu_train_batch_size 12  --max_seq_length 384   --doc_stride 128 --threads 8
```
