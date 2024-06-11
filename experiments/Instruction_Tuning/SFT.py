# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Optional, Union

import wandb
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from peft import PeftModel, FourierConfig, TaskType, get_peft_model
from datasets import load_dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<unk>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

QUESTION_PROMPT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
ANSWER_PROMPT = "### Response:\n"

@dataclass
class ModelArguments:
    model_tag: str = field(
        default="llama2",
        metadata={"help": "Model tag or path to model."},
    )
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    fourier_init: bool = field(
        default=True,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    full_precision:  bool = field(
        default=True,
        metadata={"help": "False: Use bitsandbytes Linear4bit, real quantization"
                          "True: Use quantization equivalent fp16/fp32 weights."
                  },
    )


@dataclass
class DataArguments:
    data_tag: str = field(
        default="alpaca",
        metadata={"help": "Dataset tag or path to dataset."}
    )
    data_name_or_path: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "Dataset name."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    learning_rate: float = field(default=1e-1)
    warmup_ratio: float = field(default=0.05)
    weight_decay: float = field(default=0.0)
    optim: str = field(default="adamw_torch")
    adam_epsilon: float = field(default=1e-8)
    max_grad_norm: float = field(default=1.0)
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    seed: int = field(default=0, metadata={"help": "random seed for initialization"})
    
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    
    local_rank: int = field(default=-1)
    cache_dir: Optional[str] = field(default=None)
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    n_frequency: Optional[int] = field(default=1000, metadata={"help": "the num_frequency of the Fourier adapters"})
    scale: Optional[float] = field(default=300.0, metadata={"help": "the scale of the Fourier adapters"})

    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "the evaluation strategy"})
    eval_steps: Optional[int] = field(default=5, metadata={"help": "the number of evaluation steps"})
    eval_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of evaluation accumulation steps"})

    logging_steps: int = field(default=500)
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the save strategy"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    save_steps: Optional[int] = field(
        default=50, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})

    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    mixed_precision: Optional[str] = field(default="bf16", metadata={"help": "Mixed precision training"})
    
    run_name: str = field(
        default="",
        metadata={"help": "Experiment name"},
    )
    report_to: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [QUESTION_PROMPT.format(instruction=example['instruction'], input=example['input']) for example in raw_data]
        targets = [ANSWER_PROMPT + example['output'] for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset = load_dataset(data_args.data_name_or_path, "train").train_test_split(train_size=0.8)
    dataset = load_dataset(data_args.data_name_or_path)['train'].train_test_split(train_size=0.8)
    train_set, eval_set = dataset['train'], dataset['test']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer, data_args=data_args)
    eval_dataset = SupervisedDataset(raw_data=eval_set, tokenizer=tokenizer, data_args=data_args)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    os.environ["WANDB_NAME"] = training_args.run_name

    if model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=training_args.load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
        )
    ##########################
    #       Peft Model       #
    ##########################
    if model_args.fourier_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
            # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"] 
            target_modules = ["q_proj", "v_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        training_args.target_modules = target_modules

        fourier_config = FourierConfig(
            task_type=task_type,
            inference_mode=False,
            n_frequency=training_args.n_frequency,
            scale=training_args.scale
        )
        model = get_peft_model(model, fourier_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=True,
            token=model_args.token,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder='loftq_init',
            is_trainable=True,
            token=model_args.token,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.output_dir = output_dir

        
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()