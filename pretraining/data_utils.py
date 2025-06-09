import json
import os
import random
import re
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk, concatenate_datasets
from itertools import chain
from collections import defaultdict, Counter
from tqdm import tqdm

import multiprocessing
NUM_PROC = min(32, multiprocessing.cpu_count())
WITH_INDICES = False
LOAD_FROM_CACHE_FILE = True
KEEP_IN_MEMORY = False

print(f"Using `{NUM_PROC}` processors.")

# Patterns for query languages and their names
QL_PATTERNS = [
    (r"\bSELECT\b.*\bFROM\b", "SQL"),
    # (r"\bSELECT\s+.+?\s+FROM\s+.+?(WHERE|GROUP\s+BY|ORDER\s+BY|;|$)", "SQL"),
    (r"\bMATCH\b.*\bRETURN\b", "Cypher"),
    (r"\bselect\b.*\bwhere\b", "SPARQL"),
    # (r"(db\.\w+\.[^;]+)", "MQL"),   # MQL (https://github.com/jf87/SM3-Text-to-Query/blob/main/src/evaluation/DB_query.py)
]

def load_config(ds: str) -> dict:
    cfg_mapping = {
        "openwebtext": "./configs/00_config_openwebtext.json",
        "openwebtext-10k": "./configs/00_config_openwebtext-10k.json",
        "realnewslike": "./configs/00_config_c4-realnewslike.json"
    }
    with open(cfg_mapping[ds]) as f:
        return json.load(f)

def get_dataset_len(dataset_path, split='train'):
    dataset = load_dataset(dataset_path, split=split)
    return len(dataset)

def get_chunk_ranges(total, chunk_size):
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]

def load_and_split_data(dataset: str, subdataset: str | None, cache_dir, split_fraction: str, test_size: str) -> Dataset | DatasetDict:
    dataset: Dataset | DatasetDict = load_dataset(dataset, subdataset, split=split_fraction, cache_dir=cache_dir, trust_remote_code=True, streaming=True)
    return dataset.train_test_split(test_size=test_size, writer_batch_size=100000)

def chunked_annotate_and_filter(
    dataset_path, chunk_ranges, output_dir, tokenizer, batch_size=64, num_proc=1
):
    import gc
    from datasets import load_from_disk
    os.makedirs(output_dir, exist_ok=True)
    ds = load_from_disk(dataset_path)
    for chunk_id, (start, end) in enumerate(chunk_ranges):
        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_id:05d}")
        done_path = os.path.join(chunk_dir, "DONE")
        if os.path.exists(done_path):
            print(f"Chunk {chunk_id} already processed, skipping.")
            continue
        print(f"\n==> Processing chunk {chunk_id} [{start}:{end}] ...")
        chunk_ds = ds.select(range(start, end))
        annotated = chunk_ds.map(
            lambda batch: filter_and_annotate(batch, tokenizer=tokenizer),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=False,
            keep_in_memory=False,
            desc="Annotate & count tokens"
        )
        filters = annotated["filters_triggered"]
        to_keep = []
        dirty_indices = []
        for i, f in enumerate(filters):
            if not f:
                to_keep.append(i)
            else:
                dirty_indices.append(i)
        clean = annotated.select(to_keep).map(
            keep_only_text,
            remove_columns=[c for c in annotated.column_names if c != "text"],
            load_from_cache_file=False,
            keep_in_memory=False,
        )
        dirty = annotated.select(dirty_indices).map(
            keep_only_text,
            remove_columns=[c for c in annotated.column_names if c != "text"],
            load_from_cache_file=False,
            keep_in_memory=False,
        )
        os.makedirs(chunk_dir, exist_ok=True)
        clean.save_to_disk(os.path.join(chunk_dir, "clean"))
        print(f"✅ Saved clean chunk {chunk_id}: {len(clean)} examples")
        dirty.save_to_disk(os.path.join(chunk_dir, "dirty"))
        print(f"✅ Saved dirty chunk {chunk_id}: {len(dirty)} examples")
        with open(done_path, "w") as f:
            f.write("done")
        del chunk_ds, annotated, clean, dirty
        gc.collect()

query_keywords = [
    "SELECT", "MATCH", "WHERE", "JOIN", "OPTIONAL", "FILTER",
    "RETURN", "INSERT", "DELETE", "SQL", "SPARQL", "CYPHER"
]

def filter_and_annotate(examples, tokenizer=None):
    texts = examples["text"]
    tokenized = tokenizer(texts, add_special_tokens=False, truncation=False)
    num_tokens = [len(ids) for ids in tokenized["input_ids"]]
    filters_triggered_list = []
    ql_type_list = []
    matched_substring_list = []
    keyword_counts_list = []
    for text in texts:
        filters_triggered = []
        ql_type = None
        matched_substring = None
        tokens_upper = re.findall(r'\w+', text.upper())
        token_counts = Counter(tokens_upper)
        keyword_counts = {kw: token_counts[kw] for kw in query_keywords if token_counts[kw] > 0}
        if keyword_counts:
            filters_triggered.append("keyword")
        for pattern, lang in QL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                filters_triggered.append("regex")
                ql_type = lang
                matched_substring = match.group(0)
                break
        filters_triggered_list.append(filters_triggered)
        ql_type_list.append(ql_type or "")
        matched_substring_list.append(matched_substring or "")
        keyword_counts_list.append(json.dumps(keyword_counts))
    return {
        "filters_triggered": filters_triggered_list,
        "ql_type": ql_type_list,
        "matched_substring": matched_substring_list,
        "keyword_counts": keyword_counts_list,
        "num_tokens": num_tokens,
    }

def keep_only_text(example):
    return {"text": example["text"]}

def tokenize_dataset(dataset, tokenizer, num_proc=NUM_PROC, with_indices=WITH_INDICES):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    if isinstance(dataset, dict):
        split_name = next(iter(dataset.keys()))
        columns = dataset[split_name].column_names
    else:
        columns = dataset.column_names
    return dataset.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=columns, with_indices=with_indices,
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
        keep_in_memory=KEEP_IN_MEMORY,)

def group_texts(tokenized_dataset, block_size=512, num_proc=NUM_PROC, with_indices=WITH_INDICES):
    def group_texts_fn(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        return result
    return tokenized_dataset.map(group_texts_fn, batched=True, num_proc=num_proc, with_indices=with_indices,
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
        keep_in_memory=KEEP_IN_MEMORY)

def save_dataset(dataset, path):
    dataset.save_to_disk(path)

def compute_dataset_stats(dataset, tokenizer=None, batch_size=64, num_proc=1):
    if "input_ids" in dataset.column_names:
        def batch_token_len(batch):
            return {"num_tokens": [len(ids) for ids in batch["input_ids"]]}
    elif "text" in dataset.column_names:
        def batch_token_len(batch):
            tok = tokenizer(batch["text"], add_special_tokens=False, truncation=False)
            if isinstance(tok["input_ids"][0], list):
                lengths = [len(ids) for ids in tok["input_ids"]]
            else:
                lengths = [len(tok["input_ids"])]
            return {"num_tokens": lengths}
    else:
        raise ValueError("Dataset must have either 'text' or 'input_ids' column.")
    mapped = dataset.map(
        batch_token_len,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=[],
        load_from_cache_file=False,
        keep_in_memory=False,
        desc=f"Counting tokens"
    )
    total_tokens = sum(mapped["num_tokens"])
    total_samples = len(mapped)
    return total_samples, total_tokens

def merge_and_balance_clean_dirty(
    chunks_dir, tokenizer, batch_size=64, num_proc=1, output_dir=None, seed=42, stats_json_path=None, block_size=512
):
    """
    Merges all clean/dirty chunks into two datasets, trims/supplements so both have matching token count.
    Groups the datasets into `block_size` tokens, saves merged/grouped datasets, prints & saves final stats, and deletes all chunk dirs.
    """
    from datasets import concatenate_datasets
    # 1. Collect all clean and dirty datasets from all chunks
    clean_datasets = []
    dirty_datasets = []
    chunk_paths = []
    for d in sorted(os.listdir(chunks_dir)):
        chunk_path = os.path.join(chunks_dir, d)
        if d.startswith("chunk_"):
            chunk_paths.append(chunk_path)
            clean_path = os.path.join(chunk_path, "clean")
            dirty_path = os.path.join(chunk_path, "dirty")
            if os.path.exists(clean_path):
                clean_datasets.append(load_from_disk(clean_path))
            if os.path.exists(dirty_path):
                dirty_datasets.append(load_from_disk(dirty_path))

    merged_clean = concatenate_datasets(clean_datasets) if clean_datasets else Dataset.from_dict({})
    merged_dirty = concatenate_datasets(dirty_datasets) if dirty_datasets else Dataset.from_dict({})
    print(f"Merged clean: {len(merged_clean)} examples")
    print(f"Merged dirty: {len(merged_dirty)} examples")

    # 2. Tokenize and group
    print("Tokenizing and grouping clean/dirty datasets ...")
    tokenized_clean = tokenize_dataset(merged_clean, tokenizer, num_proc=num_proc)
    grouped_clean = group_texts(tokenized_clean, block_size=block_size, num_proc=num_proc)
    tokenized_dirty = tokenize_dataset(merged_dirty, tokenizer, num_proc=num_proc)
    grouped_dirty = group_texts(tokenized_dirty, block_size=block_size, num_proc=num_proc)

    # 3. Count grouped blocks/tokens
    n_clean_blocks, clean_tokens = compute_dataset_stats(grouped_clean, tokenizer, batch_size, num_proc)
    n_dirty_blocks, dirty_tokens = compute_dataset_stats(grouped_dirty, tokenizer, batch_size, num_proc)
    print(f"Total grouped blocks: clean={n_clean_blocks}, dirty={n_dirty_blocks}")
    print(f"Total grouped tokens: clean={clean_tokens}, dirty={dirty_tokens}")

    # 4. Balance group block count (token count is ~identical per block)
    rng = np.random.default_rng(seed)
    # Helper to get total tokens in a grouped dataset
    def total_tokens(ds):
        return sum(len(ids) for ids in ds["input_ids"])
    clean_token_blocks = [len(ids) for ids in grouped_clean["input_ids"]]
    dirty_token_blocks = [len(ids) for ids in grouped_dirty["input_ids"]]
    clean_total_tokens = sum(clean_token_blocks)
    dirty_total_tokens = sum(dirty_token_blocks)

    if dirty_total_tokens < clean_total_tokens:
        print("Dirty has fewer tokens. Supplementing with additional clean blocks ...")
        # Sample from extra clean blocks until tokens match
        indices = np.arange(len(grouped_clean))
        rng.shuffle(indices)
        supplement_indices = []
        tokens_accum = dirty_total_tokens
        i = 0
        while tokens_accum < clean_total_tokens and i < len(indices):
            idx = int(indices[i])
            tokens = clean_token_blocks[idx]
            supplement_indices.append(idx)
            tokens_accum += tokens
            i += 1
        supplement_ds = grouped_clean.select(supplement_indices)
        final_dirty = concatenate_datasets([grouped_dirty, supplement_ds])
        final_clean = grouped_clean
    elif clean_total_tokens < dirty_total_tokens:
        print("Clean has fewer tokens. Subsampling dirty blocks ...")
        indices = np.arange(len(grouped_dirty))
        rng.shuffle(indices)
        keep_indices = []
        tokens_accum = 0
        i = 0
        while tokens_accum < clean_total_tokens and i < len(indices):
            idx = int(indices[i])
            tokens = dirty_token_blocks[idx]
            keep_indices.append(idx)
            tokens_accum += tokens
            i += 1
        final_dirty = grouped_dirty.select(keep_indices)
        final_clean = grouped_clean
    else:
        print("Already matched tokens.")
        final_clean = grouped_clean
        final_dirty = grouped_dirty

    # 5. Final stats (after grouping & balancing)
    n_final_clean_blocks, final_clean_tokens = compute_dataset_stats(final_clean, tokenizer, batch_size, num_proc)
    n_final_dirty_blocks, final_dirty_tokens = compute_dataset_stats(final_dirty, tokenizer, batch_size, num_proc)
    print(f"Final grouped blocks: clean={n_final_clean_blocks}, dirty={n_final_dirty_blocks}")
    print(f"Final matched tokens: clean={final_clean_tokens}, dirty={final_dirty_tokens}")

    stats = {
        "orig_clean": {
            "num_blocks": n_clean_blocks,
            "num_tokens": clean_total_tokens,
        },
        "orig_dirty": {
            "num_blocks": n_dirty_blocks,
            "num_tokens": dirty_total_tokens,
        },
        "final_clean": {
            "num_blocks": n_final_clean_blocks,
            "num_tokens": final_clean_tokens,
        },
        "final_dirtymatched": {
            "num_blocks": n_final_dirty_blocks,
            "num_tokens": final_dirty_tokens,
        },
        "match_ratio": round(final_dirty_tokens / final_clean_tokens, 6) if final_clean_tokens else None
    }

    # 6. Save grouped datasets
    if output_dir is not None:
        clean_out = os.path.join(output_dir, "final_clean_grouped")
        dirty_out = os.path.join(output_dir, "final_dirtymatched_grouped")
        final_clean.save_to_disk(clean_out)
        final_dirty.save_to_disk(dirty_out)
        print(f"Saved grouped clean: {clean_out} ({len(final_clean)} blocks)")
        print(f"Saved grouped dirtymatched: {dirty_out} ({len(final_dirty)} blocks)")

    # 7. Save stats as JSON
    stats_path = stats_json_path or os.path.join(output_dir, "final_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Final stats saved to {stats_path}")

    # 8. Remove chunk folders
    print("Deleting all chunk_* folders ...")
    for path in chunk_paths:
        try:
            import shutil
            shutil.rmtree(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")

    return final_clean, final_dirty, stats
