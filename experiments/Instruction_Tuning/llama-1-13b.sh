export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
python SFT.py \
    --model_tag llama-1-13b \
    --model_name_or_path meta/Llama-1-13b-hf \
    --n_frequency 1000 \
    --num_train_epochs 3 \
    --weight_decay 2e-2 \
    --learning_rate 0.1 \
    --scale 300.0 