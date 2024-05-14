
def get_data(dataset_name, cache_dir, local_dir=None):
    if local_dir is not None:
        import os
        from datasets import load_from_disk
        train_ds = load_from_disk(os.path.join(local_dir, dataset_name, "train"))
        val_ds = load_from_disk(os.path.join(local_dir, dataset_name, "val"))
        test_ds = load_from_disk(os.path.join(local_dir, dataset_name, "test"))
    else:
        from datasets import load_dataset
        if dataset_name == "flowers":
            train_val_ds = load_dataset("nelorth/oxford-flowers", split="train", cache_dir=cache_dir)
            test_ds = load_dataset("nelorth/oxford-flowers", split="test", cache_dir=cache_dir)
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        elif dataset_name == "dtd":
            train_val_test_ds = load_dataset("cansa/Describable-Textures-Dataset-DTD", split="train", cache_dir=cache_dir)
            splits = train_val_test_ds.train_test_split(test_size=0.2)
            train_val_ds = splits["train"]
            test_ds = splits["test"]
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        elif dataset_name == "food":
            train_val_ds = load_dataset("food101", split="train", cache_dir=cache_dir, trust_remote_code=True)
            test_ds = load_dataset("food101", split="validation", cache_dir=cache_dir, trust_remote_code=True)
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        elif dataset_name == "pets":
            train_val_ds = load_dataset("timm/oxford-iiit-pet", split="train", cache_dir=cache_dir)
            test_ds = load_dataset("timm/oxford-iiit-pet", split="test", cache_dir=cache_dir)
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        elif dataset_name == "resisc":
            train_ds = load_dataset("timm/resisc45", split="train", cache_dir=cache_dir)
            val_ds = load_dataset("timm/resisc45", split="validation", cache_dir=cache_dir)
            test_ds = load_dataset("timm/resisc45", split="test", cache_dir=cache_dir)
        elif dataset_name == "eurosat":
            train_ds = load_dataset("timm/eurosat-rgb", split="train", cache_dir=cache_dir)
            val_ds = load_dataset("timm/eurosat-rgb", split="validation", cache_dir=cache_dir)
            test_ds = load_dataset("timm/eurosat-rgb", split="test", cache_dir=cache_dir)
        elif dataset_name == "cars":
            train_val_ds = load_dataset("Multimodal-Fatima/StanfordCars_train", split="train", cache_dir=cache_dir)
            test_ds = load_dataset("Multimodal-Fatima/StanfordCars_test", split="test", cache_dir=cache_dir)
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        elif dataset_name == "fgvc":
            train_val_ds = load_dataset("Multimodal-Fatima/FGVC_Aircraft_train", split="train", cache_dir=cache_dir)
            test_ds = load_dataset("Multimodal-Fatima/FGVC_Aircraft_test", split="test", cache_dir=cache_dir)
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        elif dataset_name == "cifar10":
            train_val_ds = load_dataset("Multimodal-Fatima/CIFAR10_train", split="train", cache_dir=cache_dir)
            test_ds = load_dataset("Multimodal-Fatima/CIFAR10_test", split="test", cache_dir=cache_dir)
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        elif dataset_name == "cifar100":
            train_val_ds = load_dataset("Multimodal-Fatima/CIFAR100_train", split="train", cache_dir=cache_dir)
            test_ds = load_dataset("Multimodal-Fatima/CIFAR100_test", split="test", cache_dir=cache_dir)
            splits = train_val_ds.train_test_split(test_size=0.1)
            train_ds = splits["train"]
            val_ds = splits["test"]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds, test_ds


def main(args_out,
    model_name_or_path: str,
    dataset_name: str,
    mode: str,
    n_frequency: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    n_epoch: int,
    n_trial: int,
    results_dir: str,
    cache_dir: str,
    data_local_dir: str = None,
    
):

    import os, torch
    hub_cache_dir = os.path.join(cache_dir, "huggingface", "hub")
    data_cache_dir = os.path.join(cache_dir, "huggingface", "datasets")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = get_data(dataset_name, data_cache_dir, data_local_dir)
    labels = train_ds.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    from transformers import AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, cache_dir=hub_cache_dir)

    from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)
    test_ds.set_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    from transformers import AutoModelForImageClassification
    from utils import print_trainable_parameters

    def get_model():
        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
            cache_dir=hub_cache_dir,
        )
        print_trainable_parameters(model)

        if mode == "fourier":
            from peft import FourierConfig, FourierModel
            config = FourierConfig(
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
                n_frequency=n_frequency,
                scale=args_out.scale
            )
            model = FourierModel(model, config, 'default')
            model.set_extra_trainable(["classifier"])
        elif mode == "lora":
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            model = get_peft_model(model, config)
        elif mode == "head":
            model.requires_grad_(False)
            model.classifier.requires_grad_(True)
        elif mode == "full":
            model.requires_grad_(True)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        print_trainable_parameters(model)
        return model

    model_name = model_name_or_path.split("/")[-1]
    
    if mode == "fourier":
        save_id = f'{model_name}-{mode}-f{n_frequency}-{dataset_name}-f{10000}'
    elif mode == "lora":
        save_id = f'{model_name}-{mode}-r{lora_r}-a{lora_alpha}-d{lora_dropout}-{dataset_name}'
    elif mode in ["head", "full"]:
        save_id = f'{model_name}-{mode}-{dataset_name}'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    os.makedirs(os.path.join(results_dir, save_id), exist_ok=True)


    import numpy as np
    import evaluate

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    from transformers import TrainingArguments, Trainer

    

    def evaluate_different_seeds(head_lr,fft_lr, weight_decay,seeds=[7, 77, 777, 7777, 77777]):
        # evaluate the model on test dataset with different seeds
        test_metrics = []
        print(head_lr,fft_lr,weight_decay,)
        trainable_params, all_param = 0, 0
        for seed in seeds:
            # set the seed
            print("Using the seed:",seed)
            import random, numpy as np, torch
            random.seed(seed), np.random.seed(seed)
            torch.manual_seed(seed), torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed)
            
            model = get_model().to(device)

            if trainable_params == 0:
                trainable_params, all_param = print_trainable_parameters(model)

            from torch.optim import AdamW
            optimizer = AdamW([
                    {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": head_lr},
                    {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": fft_lr},
                ],
                weight_decay=weight_decay
                )
            
            args = TrainingArguments(
                os.path.join(results_dir, save_id, f"testrun-{seed}"),
                remove_unused_columns=False,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                per_device_eval_batch_size=batch_size,
                fp16=False,
                num_train_epochs=n_epoch,
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                push_to_hub=False,
                label_names=["labels"],
                dataloader_num_workers=4,
                save_total_limit=1,
                save_safetensors=False,
                save_only_model=True,
                seed=seed,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=image_processor,
                compute_metrics=compute_metrics,
                data_collator=collate_fn,
                optimizers=(optimizer, None),
            )

            trainer.train()
            test_metrics.append(trainer.evaluate(test_ds)["eval_accuracy"])

        return np.mean(test_metrics), np.std(test_metrics), trainable_params, all_param

    # evaluate the model on test dataset with different seeds
    test_metrics, test_metrics_std, trainable_params, all_param = evaluate_different_seeds(args_out.head_lr,args_out.fft_lr,args_out.weight_decay)

    results = {
        "trial": 'Fine-tuning with the best chosen before',
        "test_metrics": {
            "mean": test_metrics,
            "std": test_metrics_std,
        }
    }
    if mode != "head":
        results["compression"] = {
            "trainable_parameters": trainable_params,
            "all_parameters": all_param,
            "compression_ratio": 100 * trainable_params / all_param,
        }

    # save the test metrics
    import json
    with open(os.path.join(results_dir, save_id, "results.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import os
    import sys
    import numpy as npy
    from args import *
    sys.path.append("cv_experiments")

    args = get_args()

    print(args)
    main(args_out=args,
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        mode=args.mode,
        n_frequency=args.n_frequency,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.bs,
        n_epoch=args.num_epochs,
        n_trial=args.n_trial,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        data_local_dir=args.data_local_dir,

    )