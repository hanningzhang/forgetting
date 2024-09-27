#!/bin/bash
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

# model_and_tokenizer=Qwen/Qwen2-7B
#model_and_tokenizer=google/gemma-2-9b
model_and_tokenizer=meta-llama/Meta-Llama-3.1-8B
# model_and_tokenizer=openai-community/gpt2

accelerate launch --config_file fsdp_config_diff_norm.yaml python/train.py \
    --model $model_and_tokenizer \
    --tokenizer-name $model_and_tokenizer \
    --train-data data_lmflow/train_\*.json \
    --val-data data_lmflow/test.json \
    --optimizer "name=adam, lr=1e-6, weight_decay=0.0" \
    --bf16 \
    --diff_norm \
    --warmup-ratio 0.03 \
    --norm 0.5 \
    --model-type Llama \
    --pseudo_random \
    --logging_conf_file conf/common.log_conf \
    --seed 1234 \
    --max-length 1024 \
    --epoch 1 \
    --val_batch_size 2 \
    --eval_frequency 50 \
    --response_loss_only \
    --save_dir ./llama_1e-6_adam_1epoch_diffnorm50/ \
    --global_batch_size 16 \
    --lmflow-format \
    --micro_batch_size 2