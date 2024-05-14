## The GLUE benchmark (Natural Language Understanding, NLU)

You can directly run the script files corresponding to the 6 datasets in GLUE, which already include reported hyperparameters. The reported results use the median of the five seeds ($0,11111,22222,33333,44444$). If shared entry is used, the seed of the entry is unified to $2024$.

For example, CoLA+RoBERTa-Large: ```bash cola-large-best.sh```

### Reproducibility

You can view the epoch-by-epoch log file of our runs on NVIDIA GeForce RTX 4090 GPUs. However, GPU, torch version and NVIDIA version may also affect the results under the same seed. Therefore, if you encounter inconsistent results, you may consider slightly adjusting the hyperparameters, such as head_lr, fft_lr, and scale.

### GPU cost

By running the ```eval-gpu-cost.sh``` file, we evaluate the GPU cost of FourierFT. We display peak GPU memory consumptions in the table below.
Only the results of the RoBERTa-Large model are shown, and the hyperparameters are consistent with those reported in the paper.

Unit: MB
| Model 	| Methods 	| MRPC 	| CoLA 	| QNLI 	| SST-2 	| RTE 	| STS-B 	|
|:---:	|---	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| Large 	| LoRA($r=8$) 	| 8888 	| 15842 	| OOM 	| 8200 	| 23698 	| 13600 	|
| Large 	| Ours($n=1000$) 	| 8582 	| 10632 	| OOM 	| 5704 	| 21742 	| 11986 	|
