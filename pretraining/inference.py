from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk
from collators import DataCollatorForT5MLM

# Paths
model_path = "/home/vejvar-martin-nj/git/picard/results/t5/scratch/checkpoint-800"
dataset_path = "./datasets/owt-10k-clean"

# Load dataset
lm_dataset = load_from_disk(dataset_path)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Use your custom span corruption collator (critical!)
data_collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    input_length=512,
    target_length=512,
)

# Evaluation args
training_args = TrainingArguments(
    output_dir="../results/inference",
    per_device_eval_batch_size=32,
    do_eval=True,
    do_predict=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# model.eval()
# sample_input_text = "So this cat plays with rotund ball."
# input_ids = tokenizer(sample_input_text, return_tensors="pt").input_ids.to(model.device)
# generated_ids = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
# generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# print("Generated:", generated_text)


# Now evaluation works (correct labels provided!)
eval_results = trainer.evaluate()
print("Evaluation Results:")
print(eval_results)
