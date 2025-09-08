workdir=${your_work_dir}/lm-evaluation-harness
cd $workdir
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MASTER_ADDR="${CHIEF_IP:=localhost}"
MASTER_PORT=$((1 + $RANDOM % 99999))
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

model_name= # your model name
ckpt_dir=./checkpoint

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$ckpt_dir/$model_name \
    --tasks gsm8k \
    --batch_size 2 \
    2>&1 | tee -a log/$model_name.log
