CUDA_VISIBLE_DEVICES=2 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset mrpc \
    --task mrpc \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.006 \
    --fft_lr 0.05 \
    --num_epoch 30 \
    --bs 32  \
    --scale 151 \
    --seed 0 \
    --share_entry

