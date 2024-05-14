CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-large \
    --dataset mrpc \
    --task mrpc \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.003 \
    --fft_lr 0.06 \
    --num_epoch 40 \
    --bs 32  \
    --scale 99 \
    --seed 0 \
    --share_entry

