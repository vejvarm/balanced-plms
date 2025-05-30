import csv
import gc
import json
import os
import random
from transformers import AutoTokenizer
from data_utils import (
    annotate_datasetdict_parallel,
    build_dirty_matched_set,
    collect_stats_for_set,
    count_tokens,
    filter_and_collect_stats,
    keep_only_text,
    load_and_split_data,
    tokenize_dataset,
    group_texts,
    save_dataset,
    LOAD_FROM_CACHE_FILE,
    KEEP_IN_MEMORY
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
dataset_cache_path = config_args.get("dataset_cache_path", os.path.join(cache_dir, dataset_source_path+"-preproc"))  # changed "-clean" to "-preproc"
max_seq_length = int(config_args.get("max_seq_length", 512))
test_set_size = int(config_args.get("test_set_size", 10000))
model_name = config_args["model_name_or_path"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # T5 uses EOS as padding

print(f"Downloading and preprocessing dataset to: {dataset_cache_path}")

os.makedirs(dataset_cache_path, exist_ok=True)
STATS_PATH = os.path.join(dataset_cache_path, "preprocessing_stats.csv")
STATS_ROWS = []



def init_stat(stats_path=STATS_PATH):
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_type", "split", "num_documents", "num_tokens", "num_blocks",
            "percent_tokens_relative_full", "percent_docs_relative_full",
            "dirty_supplement_from_clean_docs", "dirty_supplement_from_clean_tokens"
        ])
        writer.writeheader()


def append_and_save_stat(row: dict | list[dict], stats_path=STATS_PATH):
    if isinstance(row, dict):
        row = [row]
    with open(stats_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_type", "split", "num_documents", "num_tokens", "num_blocks",
            "percent_tokens_relative_full", "percent_docs_relative_full",
            "dirty_supplement_from_clean_docs", "dirty_supplement_from_clean_tokens"
        ])
        writer.writerows(row)


### 0. Initialize stats file
init_stat()


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

# Save shared dev BEFORE any filtering, for all future experiments!
shared_dev_save_path = os.path.join(dataset_cache_path, "shared_dev")
if "test" in dataset.keys():
    os.makedirs(shared_dev_save_path, exist_ok=True)
    save_dataset(DatasetDict({"dev": dataset["test"]}), shared_dev_save_path)
    print(f"✅ Shared dev set saved at {shared_dev_save_path}")
else:
    print("Warning: No 'test' split to save as shared dev.")

for split in dataset.keys():
    n_tokens, n_docs = count_tokens(dataset[split], tokenizer)
    tokens_full_per_split[split] = n_tokens
    docs_full_per_split[split] = n_docs

append_and_save_stat(collect_stats_for_set({"test": dataset["test"]}, "full", tokens_full_per_split, docs_full_per_split, tokenizer, save=True, dataset_cache_path=dataset_cache_path))
print(f"✅ Full dataset test set prepared and saved.")

######################
### 2. FILTER & CLEAN STATS
######################
# Step 1: Annotate in parallel
annotated_dataset = annotate_datasetdict_parallel(dataset, tokenizer)

# Step 2: Filter and get stats
dataset_filtered, filter_stats, filtered_out_indices, filtered_indices_per_split = filter_and_collect_stats(
    annotated_dataset, example_cap=args.example_cap
)

del annotated_dataset
gc.collect()

dataset_filtered_clean = DatasetDict({
    k: v.map(keep_only_text, remove_columns=[col for col in v.column_names if col != "text"], load_from_cache_file=LOAD_FROM_CACHE_FILE, keep_in_memory=KEEP_IN_MEMORY)
    for k, v in dataset_filtered.items()
})

######################
### 2.5. CLEAN STATS & CLEAN TOKEN TARGET
######################
filtered_stats_dict = {}
for split in dataset_filtered_clean.keys():
    n_tokens, n_docs = count_tokens(dataset_filtered_clean[split], tokenizer)
    filtered_stats_dict[split] = {"num_tokens": n_tokens, "num_documents": n_docs}
    percent_tokens_full = n_tokens / tokens_full_per_split[split] if tokens_full_per_split[split] else ""
    percent_docs_full = n_docs / docs_full_per_split[split] if docs_full_per_split[split] else ""
clean_token_target = {split: filtered_stats_dict[split]["num_tokens"] for split in filtered_stats_dict}

# SAVE STATS FOR FULL DATASET
append_and_save_stat(collect_stats_for_set(dataset, "full", tokens_full_per_split, docs_full_per_split, tokenizer, save=False, dataset_cache_path=dataset_cache_path))
print(f"✅ Full dataset prepared and saved.")

# SAVE STATS FOR CLEAN DATASET
append_and_save_stat(collect_stats_for_set(dataset_filtered_clean, "clean", tokens_full_per_split, docs_full_per_split, tokenizer, save=True, dataset_cache_path=dataset_cache_path))
del dataset_filtered_clean
gc.collect()
print(f"✅ Clean (filtered) dataset prepared and saved.")

######################
### 3. DIRTY-MATCHED CONSTRUCTION
######################
dirty_matched, dirty_supplement_stats = build_dirty_matched_set(
    dataset, filtered_out_indices, clean_token_target, filtered_indices_per_split, seed=args.seed
)

# delete full dataset to clear memory
del dataset
gc.collect()

######################
### 4. TOKENIZE & GROUP DIRTY-MATCHED DATASET
######################
append_and_save_stat(collect_stats_for_set(dirty_matched, "dirtymatched", tokens_full_per_split, docs_full_per_split, tokenizer, dirty_supplement_stats, save=True, dataset_cache_path=dataset_cache_path))
del dirty_matched
gc.collect()
print(f"✅ Dirty-matched dataset prepared and saved.")

# Save per-filter breakdown
with open(os.path.join(dataset_cache_path, "filter_breakdown.json"), "w") as f:
    json.dump(filter_stats, f, indent=2)
print(f"✅ Filter breakdown saved.")

# Save extra supplement breakdown for dirty-matched
with open(os.path.join(dataset_cache_path, "dirty_supplement_stats.json"), "w") as f:
    json.dump(dirty_supplement_stats, f, indent=2)
print(f"✅ Dirty supplement stats breakdown saved.")