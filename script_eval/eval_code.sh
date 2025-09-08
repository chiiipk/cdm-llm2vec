work_dir=${your_work_dir}/evalplus
cd $work_dir

device_id=0,1,2,3,4,5,6,7

export CUDA_VISIBLE_DEVICES=$device_id

model_name= # your model name
ckpt_dir= # your checkpoint dir

model_path=$ckpt_dir/$model_name
evalplus.evaluate --model $model_path \
                --dataset humaneval  \
                --greedy \
                --bs 4 \
                --backend hf

evalplus.evaluate --model $model_path \
                --dataset mbpp  \
                --greedy \
                --bs 4 \
                --backend hf 

