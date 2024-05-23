python SFT.py \
    --model_tag llama-1-7b \
    --model_name_or_path meta-llama/Llama-7b-hf \
    --n_frequency 1000 \
    --num_epochs 3 \
    --weight_decay 2e-3 \
    --lr 0.1 \
    --scale 300.0 