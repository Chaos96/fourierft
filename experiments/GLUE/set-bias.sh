CUDA_VISIBLE_DEVICES=0 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset cola \
    --task cola \
    --n_frequency 1000 \
    --max_length 512 \
    --head_lr 0.007 \
    --fft_lr 0.12 \
    --num_epoch 100 \
    --bs 32  \
    --scale 50.0 \
    --seed 44444 \
    --set_bias \
    --fc 100 \
    --width 200

