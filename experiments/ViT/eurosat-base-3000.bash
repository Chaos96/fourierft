export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0 python exec.py \
    --model-name-or-path google/vit-base-patch16-224-in21k \
    --dataset-name eurosat \
    --mode fourier \
    --n_frequency 3000 \
    --num_epochs 10 \
    --n_trial 1 \
    --head_lr 8e-4 \
    --weight_decay 3e-4 \
    --fft_lr 2e-1 \
    --scale 300.0 \
    --share_entry 