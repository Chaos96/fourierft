CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-large \
    --dataset stsb \
    --task stsb \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.001 \
    --fft_lr 0.07 \
    --num_epoch 30 \
    --bs 32  \
    --scale 121 \
    --seed 0 \
    --share_entry