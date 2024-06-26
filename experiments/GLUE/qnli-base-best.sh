CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset qnli \
    --task qnli \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.1 \
    --num_epoch 40 \
    --bs 32  \
    --scale 29.0 \
    --seed 00000 \
    --share_entry