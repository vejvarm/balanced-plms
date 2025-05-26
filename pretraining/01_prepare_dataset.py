import csv
import json
import os
from transformers import AutoTokenizer
from data_utils import (
    count_tokens,
    load_and_split_data,
    filter_dataset_with_stats,
    tokenize_dataset,
    group_texts,
    save_dataset
)
import argparse

parser = argparse.ArgumentParser(description="Preprocess and filter a dataset into unbiased version.")
parser.add_argument("ds", type=str, help="Dataset to be prepared.", choices=("openwebtext", "openwebtext-10k", "realnewslike"))
parser.add_argument("--example-cap", type=int, default=20, help="Number of examples to print out during filtering.")
args = parser.parse_args()

# Load config selection
if args.ds == "openwebtext":
    cfg_path = "./configs/00_config_openwebtext.json"
elif args.ds == "openwebtext-10k":
    cfg_path = "./configs/00_config_openwebtext-10k.json"
elif args.ds == "realnewslike":
    cfg_path = "./configs/00_config_c4-realnewslike.json"
else:
    raise NotImplementedError("Supported datasets are (openwebtext, openwebtext-10k, realnewslike).")

with open(cfg_path, "r") as f:
    config_args = json.load(f)

cache_dir = config_args.get("cache_dir", "/work/datasets")
dataset_source_path = config_args.get("dataset", "stas/openwebtext-10k")
subdataset = config_args.get("subdataset", None)
dataset_cache_path = config_args.get("dataset_cache_path", os.path.join(cache_dir, dataset_source_path+"-clean"))
max_seq_length = int(config_args.get("max_seq_length", 512))
test_set_size = int(config_args.get("test_set_size", 10000))
model_name = config_args["model_name_or_path"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # T5 uses EOS as padding

print(f"Downloading and preprocessing dataset to: {dataset_cache_path}")

stats_rows = []

# 1. Load and split, stats per split before filtering
dataset = load_and_split_data(dataset_source_path, subdataset, cache_dir=cache_dir, split_fraction="train", test_size=test_set_size)
raw_stats = {}

for split in dataset.keys():
    n_tokens, n_docs = count_tokens(dataset[split], tokenizer)
    raw_stats[split] = {"num_tokens": n_tokens, "num_documents": n_docs}
    stats_rows.append({
        "stage": "raw", "split": split, "num_documents": n_docs, "num_tokens": n_tokens,
        "num_blocks": "", "tokens_dropped": "", "docs_dropped": "",
        "percent_tokens_dropped": "", "percent_docs_dropped": ""
    })

# 2. Filter, count stats per split, collect per-filter & lang breakdown
dataset_filtered, filter_stats = filter_dataset_with_stats(dataset, tokenizer, example_cap=args.example_cap)
filtered_stats = {}

for split in dataset_filtered.keys():
    n_tokens, n_docs = count_tokens(dataset_filtered[split], tokenizer)
    filtered_stats[split] = {"num_tokens": n_tokens, "num_documents": n_docs}
    n_tokens_raw = raw_stats[split]["num_tokens"]
    n_docs_raw = raw_stats[split]["num_documents"]
    stats_rows.append({
        "stage": "filtered",
        "split": split,
        "num_documents": n_docs,
        "num_tokens": n_tokens,
        "num_blocks": "",
        "tokens_dropped": n_tokens_raw - n_tokens,
        "docs_dropped": n_docs_raw - n_docs,
        "percent_tokens_dropped": (n_tokens_raw - n_tokens) / n_tokens_raw if n_tokens_raw else "",
        "percent_docs_dropped": (n_docs_raw - n_docs) / n_docs_raw if n_docs_raw else ""
    })

# 3. Tokenize, group, stats per split
tokenized = tokenize_dataset(dataset_filtered, tokenizer, num_proc=8)
grouped = group_texts(tokenized, block_size=max_seq_length, num_proc=8)

for split in grouped.keys():
    n_blocks = len(grouped[split])
    tokens_grouped = n_blocks * max_seq_length
    n_tokens_filt = filtered_stats[split]["num_tokens"]
    stats_rows.append({
        "stage": "grouped",
        "split": split,
        "num_documents": "",
        "num_tokens": tokens_grouped,
        "num_blocks": n_blocks,
        "tokens_dropped": n_tokens_filt - tokens_grouped,
        "docs_dropped": "",
        "percent_tokens_dropped": (n_tokens_filt - tokens_grouped) / n_tokens_filt if n_tokens_filt else "",
        "percent_docs_dropped": ""
    })

# 4. Save all stats
stats_path = os.path.join(dataset_cache_path, "preprocessing_stats.csv")
os.makedirs(dataset_cache_path, exist_ok=True)
fieldnames = ["stage", "split", "num_documents", "num_tokens", "num_blocks", "tokens_dropped", "docs_dropped", "percent_tokens_dropped", "percent_docs_dropped"]
with open(stats_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stats_rows)
    print(f"✅ Stats table saved at {stats_path}")
    
# Also save expanded filter stats
with open(os.path.join(dataset_cache_path, "filter_breakdown.json"), "w") as f:
    json.dump(filter_stats, f, indent=2)
    print(f"✅ Filter stats saved at {os.path.join(dataset_cache_path, 'filter_breakdown.json')}")

save_dataset(grouped, dataset_cache_path)
print(f"✅ Preprocessed dataset saved at {dataset_cache_path}")