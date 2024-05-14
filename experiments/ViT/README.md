# Command to Run

```bash
python exec.py --model-name-or-path google/vit-base-patch16-224-in21k --dataset-name fgvc --mode fourier --n-frequency 3000 
```

You can choose model and dataset according to your need. Also, if `mode` is set to `lora`, you can pass `--lora-r`, `--lora-alpha`, `--lora-dropout` to set related parameters.

This command will run a grid search with 20 trials. You can change the number of trials by passing `--n-trial` argument.