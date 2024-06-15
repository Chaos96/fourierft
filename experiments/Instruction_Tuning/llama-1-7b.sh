export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
python SFT.py \
    --model_tag llama-1-7b \
    --model_name_or_path meta/Llama-1-7b-hf \
    --n_frequency 1000 \
    --num_train_epochs 2 \
    --weight_decay 0.01 \
    --learning_rate 0.1 \
    --scale 300.0 sd
