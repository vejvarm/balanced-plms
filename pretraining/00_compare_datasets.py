import json
import os
import re
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd

from data_utils import (
    load_and_split_data,
    filter_dataset,
    tokenize_dataset,
    group_texts,
    QL_PATTERNS,
)

# Settings
STATS_DIR = "stats"
os.makedirs(STATS_DIR, exist_ok=True)

# Query pattern (same used by filter)
pattern = re.compile("|".join(QL_PATTERNS), re.IGNORECASE)
def contains_query_language(text):
    return bool(pattern.search(text))

def stat_summary(lengths):
    if not lengths:
        return {"total": 0, "mean": 0, "min": 0, "max": 0}
    return {
        "total": int(sum(lengths)),
        "mean": float(sum(lengths) / len(lengths)),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
    }

def plot_histogram(lengths, title, fname, bins=50, xlabel="Length"):
    if not lengths:
        return
    plt.figure(figsize=(6,4))
    plt.hist(lengths, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def stats_for_split(split_name, examples, tokenizer=None):
    texts = [ex['text'] for ex in tqdm(examples, ncols=100, desc=f'{split_name}: Counting chars/queries')]
    chars = [len(t) for t in texts]
    queries = [contains_query_language(t) for t in texts]
    n_samples = len(texts)
    stats = {
        "n_samples": n_samples,
        "chars": stat_summary(chars),
        "queries": {
            "n": int(sum(queries)),
            "frac": float(sum(queries) / n_samples) if n_samples else 0.0
        }
    }
    if tokenizer is not None:
        tokens_per_sample = [len(tokenizer.encode(t)) for t in tqdm(texts, ncols=100, desc=f'{split_name}: Tokenizing samples')]
        stats["tokens"] = stat_summary(tokens_per_sample)
    return stats, chars

def stats_for_tokenized_split(split_name, examples, tokenizer):
    input_ids = [ex['input_ids'] for ex in tqdm(examples, ncols=100, desc=f'{split_name}: Counting tokens')]
    lens = [len(iids) for iids in input_ids]
    n_samples = len(input_ids)
    stats = {
        "n_samples": n_samples,
        "tokens": stat_summary(lens)
    }
    # Can't count queries anymore unless we keep raw text
    return stats, lens

def stats_for_grouped_split(split_name, examples, tokenizer):
    input_ids = [ex['input_ids'] for ex in tqdm(examples, ncols=100, desc=f'{split_name}: Grouped stats')]
    lens = [len(iids) for iids in input_ids]
    n_samples = len(input_ids)
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in tqdm(input_ids, ncols=100, desc=f'{split_name}: Decoding for query-check')]
    queries = [contains_query_language(t) for t in tqdm(texts, ncols=100, desc=f'{split_name}: Counting queries in grouped')]
    stats = {
        "n_samples": n_samples,
        "tokens": stat_summary(lens),
        "queries": {
            "n": int(sum(queries)),
            "frac": float(sum(queries) / n_samples) if n_samples else 0.0
        }
    }
    return stats, lens

def save_dict_json(d, fname):
    with open(fname, 'w') as f:
        json.dump(d, f, indent=2)

def stats_dict_to_csv(stats_dict, csv_path):
    rows = []
    for split, stats in stats_dict.items():
        flat = {"split": split}
        for key, val in stats.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    flat[f"{key}_{subkey}"] = subval
            else:
                flat[key] = val
        rows.append(flat)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

def grouped_stats_from_disk(dataset_path, tokenizer, prefix="preprocessed_"):
    print(f"\n>> Loading preprocessed (cached) grouped dataset from: {dataset_path}")
    try:
        ds = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Could not load dataset at {dataset_path}: {e}")
        return None, None
    preprocessed_stats = {}
    pre_hist_files = []
    for split in tqdm(ds.keys(), desc="Preprocessed stats by split", ncols=100):
        gsplit = ds[split]
        input_ids = [ex['input_ids'] for ex in tqdm(gsplit, ncols=100, desc=f'Pre {split}: Input IDs')]
        lens = [len(iids) for iids in input_ids]
        n_samples = len(lens)
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in tqdm(input_ids, ncols=100, desc=f'Pre {split}: Decoding for query-check')]
        queries = [contains_query_language(t) for t in tqdm(texts, ncols=100, desc=f'{split}: Counting queries in cached')]
        stats = {
            "n_samples": n_samples,
            "tokens": stat_summary(lens),
            "queries": {
                "n": int(sum(queries)),
                "frac": float(sum(queries) / n_samples) if n_samples else 0.0
            }
        }
        preprocessed_stats[split] = stats
        hist_file = f"{STATS_DIR}/{prefix}{split}_tokens.png"
        plot_histogram(lens, f"Preprocessed {split} sample token length", hist_file, xlabel="Tokens")
        pre_hist_files.append(hist_file)
    save_dict_json(preprocessed_stats, f"{STATS_DIR}/{prefix}grouped_stats.json")
    stats_dict_to_csv(preprocessed_stats, f"{STATS_DIR}/{prefix}grouped_stats.csv")
    return preprocessed_stats, pre_hist_files

def compare_grouped_stats(stats1, stats2, label1="new", label2="cached"):
    for split in stats1.keys():
        s1, s2 = stats1.get(split, {}), stats2.get(split, {})
        print(f"\n=== Split: {split} ===")
        for key in ("n_samples", "tokens", "queries"):
            print(f"  {key}: {label1}={s1.get(key)} | {label2}={s2.get(key)}")
        if s1.get("n_samples") != s2.get("n_samples"):
            print(f"    ⚠️ Count mismatch: {label1}={s1.get('n_samples')}, {label2}={s2.get('n_samples')}")

# Load config
with open("./configs/00_config.json", "r") as f:
    config_args = json.load(f)
cache_dir = config_args.get("cache_dir", "/work/datasets")
dataset_source_path = config_args.get("dataset", "stas/openwebtext-10k")
subdataset = config_args.get("subdataset", None)
max_seq_length = config_args.get("max_seq_length", 512)
model_name = config_args["model_name_or_path"]
dataset_cache_path = config_args.get("dataset_cache_path", os.path.join(cache_dir, dataset_source_path + "-clean"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# =============== LOAD AND ASSESS ALREADY PREPROCESSED DATASET ===============
preprocessed_grouped_stats, pre_hist_files = grouped_stats_from_disk(dataset_cache_path, tokenizer, prefix="preprocessed_")

# -----------------[ PRINT SIDE-BY-SIDE SUMMARY ]-----------------
print("\n==== [LOADED CACHED PREPROCESSED dataset] ====")
if preprocessed_grouped_stats is not None:
    print(json.dumps(preprocessed_grouped_stats, indent=2))
else:
    print("No preprocessed stats available.")

# --- Load and split ---
print(">> Loading...") 
dataset = load_and_split_data(dataset_source_path, subdataset, cache_dir=cache_dir, split_fraction="train")

# --- Unfiltered stats ---
print(">> Raw dataset stats:")
raw_stats = {}
for split in tqdm(dataset.keys(), desc="Raw stats by split", ncols=100):
    stats, chars = stats_for_split(split, dataset[split])
    raw_stats[split] = stats
    plot_histogram(chars, f"Raw {split} sample char length", f"{STATS_DIR}/raw_{split}_charlen.png", xlabel="Chars")
save_dict_json(raw_stats, f"{STATS_DIR}/raw_stats.json")
stats_dict_to_csv(raw_stats, f"{STATS_DIR}/raw_stats.csv")

# --- Filtered stats ---
print(">> Filtering...")
dataset_filtered = filter_dataset(dataset)

print(">> Filtered dataset stats:")
filtered_stats = {}
for split in tqdm(dataset_filtered.keys(), desc="Filtered stats by split", ncols=100):
    stats, chars = stats_for_split(split, dataset_filtered[split])
    filtered_stats[split] = stats
    plot_histogram(chars, f"Filtered {split} sample char length", f"{STATS_DIR}/filtered_{split}_charlen.png", xlabel="Chars")
save_dict_json(filtered_stats, f"{STATS_DIR}/filtered_stats.json")
stats_dict_to_csv(filtered_stats, f"{STATS_DIR}/filtered_stats.csv")

# --- Tokenize, Group ---
print(">> Tokenizing and grouping...")
tokenized_stats = {}
grouped_stats = {}
for split in tqdm(dataset_filtered.keys(), desc="Tokenized/grouped stats by split", ncols=100):
    print(f" -- tokenizing {split} --")
    data = dataset_filtered[split]
    tokenized_data = tokenize_dataset({split: data}, tokenizer, num_proc=1)[split]
    stats, tokens = stats_for_tokenized_split(split, tokenized_data, tokenizer)
    tokenized_stats[split] = stats
    plot_histogram(tokens, f"Tokenized {split} sample token length", f"{STATS_DIR}/tokenized_{split}_tokens.png", xlabel="Tokens")
    # Now group
    print(f" -- grouping {split} --")
    grouped_data = group_texts({split: tokenized_data}, block_size=max_seq_length, num_proc=1)[split]
    gstats, gtokens = stats_for_grouped_split(split, grouped_data, tokenizer)
    grouped_stats[split] = gstats
    plot_histogram(gtokens, f"Grouped {split} sample token length", f"{STATS_DIR}/grouped_{split}_tokens.png", xlabel="Tokens")

save_dict_json(tokenized_stats, f"{STATS_DIR}/tokenized_stats.json")
save_dict_json(grouped_stats, f"{STATS_DIR}/grouped_stats.json")
stats_dict_to_csv(tokenized_stats, f"{STATS_DIR}/tokenized_stats.csv")
stats_dict_to_csv(grouped_stats, f"{STATS_DIR}/grouped_stats.csv")

# -- Optionally print a summary table
print("\n=== RAW ===")
print(json.dumps(raw_stats, indent=2))
print("\n=== FILTERED ===")
print(json.dumps(filtered_stats, indent=2))
print("\n=== TOKENIZED ===")
print(json.dumps(tokenized_stats, indent=2))
print("\n==== [GROUPED FILTERED dataset] ====")
print(json.dumps(grouped_stats, indent=2))
print(f"\n✅ All stats saved to ./{STATS_DIR}")

print("\n==== COMPARISON (NEW vs CACHED GROUPED) ====")
compare_grouped_stats(grouped_stats, preprocessed_grouped_stats, label1="new", label2="cached")