import json
import pathlib
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
from collators import DataCollatorForT5MLM

# To resume from checkpoint with PyTorch > 2.6
# Allow torch.load to unpickle numpy._core.multiarray._reconstruct
from numpy._core.multiarray import _reconstruct
from numpy import ndarray
torch.serialization.add_safe_globals([_reconstruct, ndarray])

import argparse

parser = argparse.ArgumentParser(description="Preprocess and filter a dataset into unbiased verison.")
parser.add_argument("ds", type=str, help="Dataset to be prepared.", choices=("openwebtext", "openwebtext-10k", "realnewslike"))
args = parser.parse_args()

if args.ds == "openwebtext":
    cfg_path = "./configs/00_config_openwebtext.json"
elif args.ds == "openwebtext-10k":
    cfg_path = "./configs/00_config_openwebtext-10k.json"
elif args.ds == "realnewslike":
    cfg_path = "./configs/00_config_c4-realnewslike.json"
else:
    raise NotImplementedError("Supported datasets are (openwebtext, openwebtext-10k, realnewslike).")

# Load configuration from JSON file.
with open(cfg_path, "r") as f:
    config_args = json.load(f)

max_seq_length = config_args.get("max_seq_length", 512)

# 1. Load preprocessed dataset from disk
data_dir = pathlib.Path(config_args.get("dataset_cache_path", "/work/datasets/owt-10k-clean"))
variant = config_args["dataset_variant"]
train_path = data_dir.joinpath(variant)
dev_path = data_dir.joinpath("shared_dev_grouped", "grouped")
print(f"Loading dataset from train_path: `{train_path}` | dev_path: `{dev_path}`")
train_dataset = load_from_disk(str(train_path))
print(f"Train dataset: {train_dataset}")
eval_dataset = load_from_disk(str(dev_path))
print(f"Eval (dev) dataset: {eval_dataset}")

# 2. Tokenizer initialization.
tokenizer = AutoTokenizer.from_pretrained(config_args["model_name_or_path"])
tokenizer.pad_token = tokenizer.eos_token  # T5 often uses EOS as PAD.


# 3. Prepare the custom data collator.
data_collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=config_args.get("mlm_probability", 0.15),
    mean_noise_span_length=config_args.get("mean_noise_span_length", 3.0),
    input_length=max_seq_length,
    target_length=max_seq_length,  # Adjust if needed.
)

# 4. Load model configuration and model.
config = AutoConfig.from_pretrained(config_args["model_name_or_path"])
model = T5ForConditionalGeneration(config)

out_dir = pathlib.Path(config_args["output_dir"]).joinpath(variant)

# 5. Set up TrainingArguments.
training_args = TrainingArguments(
    output_dir=str(out_dir),
    do_train=config_args.get("do_train", True),
    do_eval=config_args.get("do_eval", True),
    num_train_epochs=config_args.get("num_train_epochs", 4),
    per_device_train_batch_size=config_args.get("per_device_train_batch_size", 12),
    per_device_eval_batch_size=config_args.get("per_device_eval_batch_size", 16),
    gradient_accumulation_steps=config_args.get("gradient_accumulation_steps", 4),
    learning_rate=config_args.get("learning_rate", 3e-4),
    lr_scheduler_type=config_args.get("lr_scheduler_type", "constant"),
    warmup_steps=config_args.get("num_warmup_steps", 0),
    weight_decay=config_args.get("weight_decay", 0.01),
    logging_steps=config_args.get("logging_steps", 40),
    eval_steps=config_args.get("eval_steps", 200),
    save_steps=config_args.get("save_steps", 400),
    overwrite_output_dir=config_args.get("overwrite_output_dir", False),
    resume_from_checkpoint=config_args.get("resume_from_checkpoint", False),
    ignore_data_skip=config_args.get("ignore_data_skip", False),
    save_total_limit=config_args.get("save_total_limit", 2),
    seed=config_args.get("seed", 42),
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=config_args.get("fp16", False),  # Recommended if GPU supports mixed precision
    logging_dir="../logs",
    report_to=["wandb"],
)

# 6. Create the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 7. Start training.
if training_args.do_train:
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
