#!/bin/bash

set -ex
work_dir=${your_work_dir}
cd $work_dir


export MASTER_ADDR="${CHIEF_IP:=localhost}"
MASTER_PORT=$((1 + $RANDOM % 99999))

gpu_num=8
accum_num=1
if [ $gpu_num -eq 1 ]; then
    accum_num=2
    export CUDA_VISIBLE_DEVICES=0
elif [ $gpu_num -eq 2 ]; then
    accum_num=16
    export CUDA_VISIBLE_DEVICES=0,1
elif [ $gpu_num -eq 4 ]; then
    accum_num=8
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ $gpu_num -eq 8 ]; then
    accum_num=4
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ $gpu_num -eq 16 ]; then
    accum_num=2
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
else
    echo "Unsupported GPU number: $gpu_num"
fi

model_name=qwen_dolly-sft

train_path=$work_dir/src/run_clm_sft.py
premodel=$work_dir/pretrained_LM/Qwen2-0.5B
save_dir=$work_dir
model_save=$save_dir/checkpoint/$model_name
LOG_FILE=${save_dir}/finetune/log.${model_name}
cache_dir=$work_dir/cache
export TRITON_CACHE_DIR="$cache_dir/triton"
export TORCH_HOME="$cache_dir/torch"
export TORCH_EXTENSIONS_DIR="$cache_dir/torch_extensions"
export HF_HOME="$cache_dir/huggingface"
export HF_HUB_CACHE="$cache_dir/hub"
export TRANSFORMERS_CACHE="$cache_dir/huggingface"
export CXX=g++
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export OMP_NUM_THREADS=20
TOKENIZERS_PARALLELISM=false
# HOST_NUM will be 1
HOST_NUM=1
INDEX=0
train_files=$work_dir/dolly_train_hf.json

torchrun  --nproc_per_node=$gpu_num \
    --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT $train_path \
    --model_name_or_path $premodel \
    --train_file $train_files \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $accum_num \
    --num_train_epochs 10 \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 3 \
    --do_train \
    --evaluation_strategy "no" \
    --ddp_timeout 36000 \
    --seed 42 \
    --weight_decay 0.0 \
    --adam_beta2 0.95 \
    --learning_rate 2e-5 \
    --block_size 2048 \
    --output_dir ${model_save} \
    --deepspeed config/deepspeed_config.json \
    --overwrite_output_dir \
    --overwrite_cache \
    --bf16 \
    --gradient_checkpointing True \
     2>&1 |tee ${LOG_FILE}

