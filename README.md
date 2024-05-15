This is the official implementation for the following paper:

[Parameter-Efficient Fine-Tuning with Discrete Fourier Transform](https://arxiv.org/abs/2405.03003)

*Ziqi Gao^, Qichao Wang^, Aochuan Chen^, Zijing Liu, Bingzhe Wu, Liang Chen, Jia Li**

*ICML 2024*

> The `peft` folder is a fork from huggingface's repository, which can be found [peft](https://github.com/huggingface/peft).

## Update progress
NLU (GLUE) and image classification parts have been added.

## Dependencies

1. ```cd ./peft```
2. ```pip install -e .```
3. ```pip install datasets scipy scikit-learn evaluate pillow torchvision optuna```

## Run

Please kindly move to ```./experiments``` for the specific tasks.
