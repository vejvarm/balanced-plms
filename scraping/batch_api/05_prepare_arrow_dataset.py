import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from pretraining.data_utils import tokenize_dataset, group_texts, compute_dataset_stats


def build_text(example):
    context = example.get("context")
    if context:
        text = f"{example['query']}\n{context}"
    else:
        text = example['query']
    return {"text": text, "target": example.get("target", "")}


def main():
    parser = argparse.ArgumentParser(description="Tokenize and group SPARQL explanation data")
    parser.add_argument("--input-file", default="sparql_explain_data.jsonl")
    parser.add_argument(
        "--output-dir",
        default="sparql_explain_arrow",
        help="Directory where the arrow dataset will be saved"
    )
    parser.add_argument("--model-name", default="google-t5/t5-base")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dev-size",
        type=int,
        default=100,
        help="Number of samples to hold out for validation"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]
    dataset = dataset.map(build_text, remove_columns=dataset.column_names)

    split = dataset.train_test_split(test_size=args.dev_size, seed=42)
    train_ds = split["train"]
    dev_ds = split["test"]

    tokenized_train = tokenize_dataset(train_ds, tokenizer, num_proc=args.num_proc)
    grouped_train = group_texts(tokenized_train, block_size=args.block_size, num_proc=args.num_proc)

    tokenized_dev = tokenize_dataset(dev_ds, tokenizer, num_proc=args.num_proc)
    grouped_dev = group_texts(tokenized_dev, block_size=args.block_size, num_proc=args.num_proc)

    train_samples, train_tokens = compute_dataset_stats(grouped_train, tokenizer, args.batch_size, args.num_proc)
    dev_samples, dev_tokens = compute_dataset_stats(grouped_dev, tokenizer, args.batch_size, args.num_proc)
    print(f"Train grouped dataset: blocks={train_samples}, tokens={train_tokens}")
    print(f"Dev grouped dataset: blocks={dev_samples}, tokens={dev_tokens}")

    os.makedirs(args.output_dir, exist_ok=True)
    grouped_train.save_to_disk(os.path.join(args.output_dir, "grouped"))
    dev_dir = os.path.join(args.output_dir, "shared_dev_grouped")
    os.makedirs(dev_dir, exist_ok=True)
    grouped_dev.save_to_disk(os.path.join(dev_dir, "grouped"))
    print(f"Saved train dataset to {args.output_dir}/grouped")
    print(f"Saved dev dataset to {dev_dir}/grouped")


if __name__ == "__main__":
    main()
