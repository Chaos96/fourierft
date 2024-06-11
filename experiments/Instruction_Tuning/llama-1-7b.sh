export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=3,4 python SFT.py \
    --model_tag llama-1-7b \
    --model_name_or_path /data1/keyi/model/meta-llama/Llama-1-7b-hf \
    --n_frequency 1000 \
    --num_train_epochs 3 \
    --weight_decay 2e-3 \
    --learning_rate 0.1 \
    --scale 300.0 
