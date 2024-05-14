import argparse
import os

print('Parsing args')

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
parser.add_argument("--dataset", type=str, default="mrpc")
parser.add_argument("--task", type=str, default="mrpc")
parser.add_argument("--bs", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--n_frequency", type=int, default=200)
parser.add_argument("--head_lr", type=float, default=5e-3)
parser.add_argument("--fft_lr", type=float, default=1e-1)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warm_step", type=float, default=0.06)
parser.add_argument("--train_ratio", type=float, default=1)
parser.add_argument("--scale", type=float, default=300.)
parser.add_argument("--width", type=float, default=200.)
parser.add_argument("--fc", type=float, default=1.)
parser.add_argument("--share_entry", action= "store_true")
parser.add_argument("--use_abs", action= "store_true")
parser.add_argument("--set_bias", action= "store_true")
parser.add_argument("--seed", type=int, default=00000)
parser.add_argument("--entry_seed", type=int, default=2024)

## For ViT computer vision tasks
parser.add_argument("--model-name-or-path", type=str,
                    required=True,
                    choices=[
                        "google/vit-base-patch16-224-in21k",
                        "google/vit-large-patch16-224-in21k",
                        "google/vit-huge-patch14-224-in21k",
                    ])
parser.add_argument("--dataset-name", type=str,
                    required=True,
                    choices=[
                        "flowers",
                        "pets",
                        "dtd",
                        "food",
                        "resisc",
                        "eurosat",
                        "cars",
                        "fgvc",
                        "cifar10",
                        "cifar100",
                    ])

parser.add_argument("--mode", type=str, choices=["fourier", "lora", "head", "full"])

parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=16)
parser.add_argument("--lora-dropout", type=float, default=0.1)

parser.add_argument("--n_trial", type=int, default=1)

parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--cache-dir", type=str, default=os.path.join(os.getenv("HOME"), ".cache"))
parser.add_argument("--data-local-dir", type=str, default=None)

args = parser.parse_args()

def get_args():
    return parser.parse_args()