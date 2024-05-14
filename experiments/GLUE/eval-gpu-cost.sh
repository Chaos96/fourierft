CUDA_VISIBLE_DEVICES=2 python NLU_GLUE.py \
    --model_name_or_path roberta-base \
    --dataset mrpc \
    --task mrpc \
    --n_frequency 1000 \
    --max_length 512 \
    --num_epoch 100 \
    --bs 32  \
    --seed 0 \

