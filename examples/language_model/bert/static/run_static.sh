#!/bin/bash
#export FLAGS_USE_STANDALONE_EXECUTOR=1

unset CUDA_VISIBLE_DEVICES
python3 -m paddle.distributed.launch --gpus "3" run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --max_predictions_per_seq 76 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir data/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --device gpu \
    --use_amp true 
