CUDA_VISIBLE_DEVICES=2 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset cola \
    --task cola \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.008 \
    --fft_lr 0.12 \
    --num_epoch 100 \
    --bs 32  \
    --scale 49.0 \
    --seed 0 \
    --share_entry

