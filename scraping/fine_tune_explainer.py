from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# 1. Load your dataset
dataset = load_dataset("json", data_files={"train": "sparql_explain_data.jsonl"})["train"]
# Optionally split (if you have human-labeled 'target' for eval)
dataset = dataset.train_test_split(test_size=0.1, seed=42) if "target" in dataset.features else {"train": dataset}

# 2. Model & tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Preprocessing
def preprocess(example):
    model_inputs = tokenizer(
        example["input"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    if example["target"]:
        labels = tokenizer(
            example["target"],
            truncation=True,
            max_length=128,
            padding="max_length"
        ).input_ids
        # Label padding: HF expects -100 for ignore
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
        model_inputs["labels"] = labels
    return model_inputs

tokenized = dataset.map(preprocess, batched=False, remove_columns=dataset["train"].column_names)

# 4. Training setup
training_args = TrainingArguments(
    output_dir="flant5-sparql-explainer",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=50,
    evaluation_strategy="steps" if "test" in tokenized else "no",
    eval_steps=200 if "test" in tokenized else None,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    seed=42,
    report_to="none"
)
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("test"),
    data_collator=collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("flant5-sparql-explainer")
tokenizer.save_pretrained("flant5-sparql-explainer")
