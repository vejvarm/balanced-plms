import csv
import json
import os
import random
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
from datasets import DatasetDict

parser = argparse.ArgumentParser(description="Prepare, filter, and analyze datasets (full, clean, dirty).")
parser.add_argument("ds", type=str, help="Dataset to be prepared.", choices=("openwebtext", "openwebtext-10k", "realnewslike"))
parser.add_argument("--example-cap", type=int, default=20, help="Number of example dirty texts to save for each trigger/lang combo.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Load config selection
cfg_mapping = {
    "openwebtext": "./configs/00_config_openwebtext.json",
    "openwebtext-10k": "./configs/00_config_openwebtext-10k.json",
    "realnewslike": "./configs/00_config_c4-realnewslike.json"
}
cfg_path = cfg_mapping[args.ds]

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

######################
### 1. LOAD & RAW STATS
######################
dataset = load_and_split_data(
    dataset_source_path, subdataset, cache_dir=cache_dir, 
    split_fraction="train", test_size=test_set_size
)
raw_stats = {}
tokens_full_per_split = {}
docs_full_per_split = {}

for split in dataset.keys():
    n_tokens, n_docs = count_tokens(dataset[split], tokenizer)
    tokens_full_per_split[split] = n_tokens
    docs_full_per_split[split] = n_docs
    stats_rows.append({
        "dataset_type": "full", "split": split, 
        "num_documents": n_docs, "num_tokens": n_tokens, "num_blocks": "",
        "percent_tokens_relative_full": 1.0,
        "percent_docs_relative_full": 1.0,
        "dirty_supplement_from_clean_docs": "",
        "dirty_supplement_from_clean_tokens": ""
    })

# Save shared dev split before filtering
shared_dev_save_path = dataset_cache_path + "_shared_dev"
if "test" in dataset.keys():
    os.makedirs(os.path.dirname(shared_dev_save_path), exist_ok=True)
    save_dataset(DatasetDict({"dev": dataset["test"]}), shared_dev_save_path)
    print(f"✅ Shared dev set saved at {shared_dev_save_path}")

######################
### 2. FILTER & CLEAN STATS
######################
dataset_filtered, filter_stats, filtered_out_indices, filtered_indices_per_split = \
    filter_dataset_with_stats(dataset, tokenizer, example_cap=args.example_cap)
filtered_stats_dict = {}

for split in dataset_filtered.keys():
    n_tokens, n_docs = count_tokens(dataset_filtered[split], tokenizer)
    filtered_stats_dict[split] = {"num_tokens": n_tokens, "num_documents": n_docs}
    percent_tokens_full = n_tokens / tokens_full_per_split[split] if tokens_full_per_split[split] else ""
    percent_docs_full = n_docs / docs_full_per_split[split] if docs_full_per_split[split] else ""
    stats_rows.append({
        "dataset_type": "clean", "split": split, 
        "num_documents": n_docs, "num_tokens": n_tokens, "num_blocks": "",
        "percent_tokens_relative_full": percent_tokens_full,
        "percent_docs_relative_full": percent_docs_full,
        "dirty_supplement_from_clean_docs": "",
        "dirty_supplement_from_clean_tokens": ""
    })
clean_token_target = {split: filtered_stats_dict[split]["num_tokens"] for split in filtered_stats_dict}

######################
### 3. DIRTY-MATCHED CONSTRUCTION
######################

def build_dirty_matched_set(dataset, filtered_out_indices, clean_token_target, filtered_indices_per_split, seed=42):
    random.seed(seed)
    dirty_dict = DatasetDict()
    supplement_stats_per_split = {}
    for split in filtered_out_indices:
        data = dataset[split]
        dirty_info = filtered_out_indices[split]
        random.shuffle(dirty_info)
        selected_dirty = []
        total_tokens = 0
        for entry in dirty_info:
            if total_tokens >= clean_token_target[split]:
                break
            selected_dirty.append(entry["idx"])
            total_tokens += entry["tokens"]
        # Supplement with random clean if needed
        n_tokens_dirty = total_tokens
        n_docs_dirty = len(selected_dirty)
        supplement_indices = []
        supplement_tokens = 0
        supplement_docs = 0
        if total_tokens < clean_token_target[split]:
            needed = clean_token_target[split] - total_tokens
            all_clean_indices = set(filtered_indices_per_split[split])
            dirty_indices_set = set(e["idx"] for e in dirty_info)
            eligible_clean_indices = list(all_clean_indices - dirty_indices_set)
            random.shuffle(eligible_clean_indices)
            for i in eligible_clean_indices:
                ex_len = len(data[i]["text"])
                supplement_indices.append(i)
                supplement_tokens += len(data[i]["text"])
                supplement_docs += 1
                if total_tokens + supplement_tokens >= clean_token_target[split]:
                    break
            all_indices = selected_dirty + supplement_indices
        else:
            supplement_tokens = 0
            supplement_docs = 0
            all_indices = selected_dirty
        supplement_stats_per_split[split] = {
            "n_docs_dirty": n_docs_dirty, 
            "n_tokens_dirty": n_tokens_dirty,
            "n_docs_supplement": supplement_docs, 
            "n_tokens_supplement": supplement_tokens,
            "n_total": len(all_indices)
        }
        dirty_dict[split] = data.select(all_indices)
    return dirty_dict, supplement_stats_per_split

dirty_matched, dirty_supplement_stats = build_dirty_matched_set(
    dataset, filtered_out_indices, clean_token_target, filtered_indices_per_split, seed=args.seed
)

######################
### 4. TOKENIZE & GROUP ALL THREE DATASETS
######################
def collect_stats_for_set(dataset_dict, dataset_type, ref_tokens_per_split, ref_docs_per_split, supplement_stats=None):
    rows = []
    for split in dataset_dict.keys():
        n_docs = len(dataset_dict[split])
        n_tokens, _ = count_tokens(dataset_dict[split], tokenizer)
        # Get number of group blocks
        tknz = tokenize_dataset(DatasetDict({split: dataset_dict[split]}), tokenizer, num_proc=8)
        grouped = group_texts(tknz, block_size=max_seq_length, num_proc=8)
        n_blocks = len(grouped[split])
        row = {
            "dataset_type": dataset_type,
            "split": split, 
            "num_documents": n_docs,
            "num_tokens": n_tokens,
            "num_blocks": n_blocks,
            "percent_tokens_relative_full": n_tokens / ref_tokens_per_split[split] if ref_tokens_per_split[split] else "",
            "percent_docs_relative_full": n_docs / ref_docs_per_split[split] if ref_docs_per_split[split] else "",
            "dirty_supplement_from_clean_docs": supplement_stats[split]['n_docs_supplement'] if supplement_stats and split in supplement_stats else "",
            "dirty_supplement_from_clean_tokens": supplement_stats[split]['n_tokens_supplement'] if supplement_stats and split in supplement_stats else ""
        }
        rows.append(row)
    return rows

# Tokenize/group/output for full, clean (filtered), and dirty matched
stats_rows += collect_stats_for_set(dataset, "full", tokens_full_per_split, docs_full_per_split)
# For clean, use the filtered dataset
stats_rows += collect_stats_for_set(dataset_filtered, "clean", tokens_full_per_split, docs_full_per_split)
# For dirty, use the dirty (composed) dataset and include supplement stats
stats_rows += collect_stats_for_set(dirty_matched, "dirty", tokens_full_per_split, docs_full_per_split, dirty_supplement_stats)

# Save datasets (grouped set, clean grouped, dirty-matched grouped)
# Saving the grouped versions:
for ds_type, ds, suffix in [
    ("filtered", dataset_filtered, "_clean"),
    ("dirtymatched", dirty_matched, "_dirtymatched"),
]:
    tokenized = tokenize_dataset(ds, tokenizer, num_proc=8)
    grouped = group_texts(tokenized, block_size=max_seq_length, num_proc=8)
    save_dataset(grouped, dataset_cache_path + suffix)

print(f"✅ Clean (filtered) and dirty-matched datasets prepared and saved.")

######################
### 5. SAVE STATS AND BREAKDOWN
######################
stats_path = os.path.join(dataset_cache_path, "preprocessing_stats.csv")
os.makedirs(dataset_cache_path, exist_ok=True)
fieldnames = [
    "dataset_type", "split", "num_documents", "num_tokens", "num_blocks", 
    "percent_tokens_relative_full", "percent_docs_relative_full",
    "dirty_supplement_from_clean_docs", "dirty_supplement_from_clean_tokens"
]
with open(stats_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stats_rows)
print(f"✅ Stats table saved at {stats_path}")

# Save per-filter breakdown
with open(os.path.join(dataset_cache_path, "filter_breakdown.json"), "w") as f:
    json.dump(filter_stats, f, indent=2)
print(f"✅ Filter breakdown saved.")

# Save extra supplement breakdown for dirty-matched
with open(os.path.join(dataset_cache_path, "dirty_supplement_stats.json"), "w") as f:
    json.dump(dirty_supplement_stats, f, indent=2)
print(f"✅ Dirty supplement stats breakdown saved.")