import argparse
import os
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    FourierConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    FourierModel
)
import optuna
import evaluate
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm


def objective(trial): 
    scale = trial.suggest_float("scale",50,350)
    head_lr = trial.suggest_float("head_lr",5e-4,5e-2)
    lora_lr = trial.suggest_float("lora_lr",1e-2,2e-1)
    print('scale=',scale,';','head_lr=',head_lr,';','lora_lr=',lora_lr)
    n_frequency = 1000
    weight_decay = 0.
    model_name_or_path = "roberta-large"
    task = "cola"
    peft_type = PeftType.FOURIER
    device = "cuda"
    num_epochs = 30
    num_labels = 2
    max_length = 128
    batch_size = 32
    if task == "stsb":
        num_labels = 1
    peft_config = FourierConfig(task_type="SEQ_CLS", inference_mode=False, n_frequency = n_frequency, scale = scale)

    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", task)
    # datasets = load_dataset(dataset)
    # datasets['train'] = datasets['train'].select(range(int(len(datasets['train']) * 0.1)))
    # metric = evaluate.load("glue", task)
    metric = load_metric("glue", task)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        if task == 'sst2' or task == 'cola':
            outputs = tokenizer(examples["sentence"], truncation=True, max_length=max_length)
        elif task == 'qnli':
            outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=max_length)
        elif task == 'qqp':
            outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=max_length)
        else:
            outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
        return outputs

    if task == 'sst2' or task == 'cola':
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence"],
        )
    elif task == 'qnli':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question", "sentence"],
        )
    elif task == 'qqp':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question1", "question2"],
        )
    else:
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
        )
    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")


    # Instantiate dataloaders.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    print("now seed model:",torch.initial_seed())
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,num_labels=num_labels,return_dict=True)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model
    # for param in model.classifier.parameters():
    #     param.requires_grad = False

    head_param = list(map(id, model.classifier.parameters()))

    others_param = filter(lambda p: id(p) not in head_param, model.parameters()) 

    # optimizer = AdamW(params=model.parameters(), lr=lr)



    model.to(device)
    acc_list = []
    
    
    optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": head_lr},
    {"params": others_param, "lr": lora_lr}
    ],weight_decay=weight_decay
    )
# Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            if task == "stsb":
                predictions = outputs.logits
            else:
                predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        if task == "stsb":
            acc_list.append(eval_metric['pearson'])
            print(f"epoch {epoch}:", eval_metric, ', current_best_pearson:',max(acc_list),'train_loss:',loss)
        elif task == 'cola':
            acc_list.append(eval_metric['matthews_correlation'])
            print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list),'train_loss:',loss)
        else:
            acc_list.append(eval_metric['accuracy'])
            print(f"epoch {epoch}:", eval_metric, ', current_best_acc:',max(acc_list),'train_loss:',loss)
    
    return max(acc_list)

study = optuna.create_study(direction='maximize',
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        )
study.optimize(objective, n_trials=100)
best_trial = study.best_trial
print("Best hyperparameters: {}".format(best_trial.params))