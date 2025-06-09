import os
import json
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from data_utils import (
    get_chunk_ranges, chunked_annotate_and_filter,
    merge_and_balance_clean_dirty, tokenize_dataset, group_texts, compute_dataset_stats
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str, choices=("openwebtext", "openwebtext-10k", "realnewslike"))
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--max_chunks", type=int, default=None)
    args = parser.parse_args()

    cfg_mapping = {
        "openwebtext": "./configs/00_config_openwebtext.json",
        "openwebtext-10k": "./configs/00_config_openwebtext-10k.json",
        "realnewslike": "./configs/00_config_c4-realnewslike.json"
    }
    with open(cfg_mapping[args.ds]) as f:
        config_args = json.load(f)

    dataset_path = config_args.get("dataset", "stas/openwebtext-10k")
    cache_dir = config_args.get("cache_dir", "./datasets")
    dataset_cache_path = config_args.get("dataset_cache_path", f"./{args.ds}-preproc-chunks")
    model_name = config_args["model_name_or_path"]
    max_seq_length = int(config_args.get("max_seq_length", 512))
    test_set_size = int(config_args.get("test_set_size", 10000))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    shared_dev_dir = os.path.join(dataset_cache_path, "shared_dev_grouped")
    raw_train_path = os.path.join(dataset_cache_path, "raw_train")

    # === 1. Split dataset only if necessary ===
    if not (os.path.exists(os.path.join(shared_dev_dir, "grouped")) and os.path.exists(raw_train_path)):
        print("Splitting dataset into train/dev splits...")
        full_dataset = load_dataset(dataset_path, split="train", streaming=False, cache_dir=cache_dir)
        split = full_dataset.train_test_split(test_size=test_set_size, seed=42)
        raw_train = split["train"]
        raw_dev = split["test"]

        raw_train.save_to_disk(raw_train_path)
        print(f"Saved raw_train at {raw_train_path}, {len(raw_train)} samples")

        os.makedirs(shared_dev_dir, exist_ok=True)
        tokenized_dev = tokenize_dataset(raw_dev, tokenizer, num_proc=args.num_proc)
        grouped_dev = group_texts(tokenized_dev, block_size=max_seq_length, num_proc=args.num_proc)
        grouped_dev.save_to_disk(os.path.join(shared_dev_dir, "grouped"))
        print(f"Saved grouped dev set at {shared_dev_dir}/grouped, {len(grouped_dev)} blocks")

        del full_dataset, split, raw_train, raw_dev, tokenized_dev, grouped_dev
        import gc; gc.collect()
    else:
        print("Grouped dev set and raw_train already exist, loading from disk.")

    # === 2. Load grouped dev and raw_train from disk for further processing ===
    grouped_dev = load_from_disk(os.path.join(shared_dev_dir, "grouped"))
    raw_train = load_from_disk(raw_train_path)

    # === 3. Compute and save stats for grouped dev set ===
    shared_dev_stats_file = os.path.join(shared_dev_dir, "dev_stats.json")
    dev_samples, dev_tokens = compute_dataset_stats(grouped_dev, tokenizer, batch_size=args.batch_size, num_proc=args.num_proc)
    shared_stats = {
        "split": "dev",
        "grouped_num_blocks": dev_samples,
        "grouped_num_tokens": dev_tokens,
    }
    with open(shared_dev_stats_file, "w") as f:
        json.dump(shared_stats, f, indent=2)
    print(f"Dev set (grouped) stats: {shared_stats}")

    # === 4. Chunked annotation/filtering for train split ===
    total = len(raw_train)
    chunk_ranges = get_chunk_ranges(total, args.chunk_size)
    if args.max_chunks is not None:
        chunk_ranges = chunk_ranges[:args.max_chunks]
    print(f"Total train samples: {total}, Num chunks: {len(chunk_ranges)}")
    chunked_annotate_and_filter(
        raw_train_path, chunk_ranges, dataset_cache_path, tokenizer,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
    )

    # === 5. Merge and balance, compute and save stats, delete chunk dirs ===
    merge_and_balance_clean_dirty(
        chunks_dir=dataset_cache_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        output_dir=dataset_cache_path,
        seed=42,
        stats_json_path=os.path.join(dataset_cache_path, "final_stats.json"),
        block_size=max_seq_length
    )

    print("\nAll processing finished. Training set chunked, cleaned/dirty sets balanced, dev set saved.")

if __name__ == "__main__":
    main()
