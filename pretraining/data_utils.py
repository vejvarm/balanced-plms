import json
import os
import random
import re
from datasets import load_dataset, DatasetDict, Dataset
from itertools import chain
from collections import defaultdict, Counter
from tqdm import tqdm

import multiprocessing
NUM_PROC = min(32, multiprocessing.cpu_count())
WITH_INDICES = False
LOAD_FROM_CACHE_FILE = False
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

def load_and_split_data(dataset: str, subdataset: str | None, cache_dir, split_fraction: str, test_size: str) -> Dataset | DatasetDict:
    dataset: Dataset | DatasetDict = load_dataset(dataset, subdataset, split=split_fraction, cache_dir=cache_dir, trust_remote_code=True)
    return dataset.train_test_split(test_size=test_size, writer_batch_size=100000)


# Keyword filter (count tokens matched)
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



def filter_and_collect_stats(annotated_ds, example_cap=10):
    """
    Filters the dataset and simultaneously collects filter statistics.
    Returns: filtered_dataset (matching original HF DatasetDict), filter_stats dictionary
    """
    if isinstance(annotated_ds, DatasetDict):
        result_dict = {}
        stats_dict = defaultdict(lambda: defaultdict(lambda: {"count": 0, "tokens": 0, "examples": [], "keyword_counts": {k:0 for k in query_keywords}}))
        filtered_indices_per_split = {}
        filtered_out_indices = defaultdict(list)
        for split, ds in annotated_ds.items():
            keep_indices = []
            # Convert columns to list once for fast access
            filters_triggered_list = ds["filters_triggered"]
            ql_type_list = ds["ql_type"]
            matched_substring_list = ds["matched_substring"]
            keyword_counts_list = ds["keyword_counts"]
            num_tokens_list = ds["num_tokens"]
            text_list = ds["text"]
            for i in tqdm(range(len(ds)), desc=f"Filtering/stats for `{split}`"):
                filters_trig = filters_triggered_list[i]
                ql_type = ql_type_list[i] or "unknown"
                matched_substring = matched_substring_list[i]
                keyword_counts = json.loads(keyword_counts_list[i])
                ex_tokens = num_tokens_list[i]
                if not filters_trig:
                    keep_indices.append(i)
                else:
                    filtered_out_indices[split].append({
                        "idx": i,
                        "tokens": ex_tokens,
                        "filters": filters_trig,
                        "lang": ql_type,
                    })
                    for fil in filters_trig:
                        fs = stats_dict[fil][ql_type]
                        fs["count"] += 1
                        fs["tokens"] += ex_tokens
                        for k in query_keywords:
                            fs["keyword_counts"][k] += keyword_counts.get(k, 0)
                        if len(fs["examples"]) < example_cap:
                            fs["examples"].append({
                                "text": text_list[i],
                                "matched": matched_substring
                            })
            result_dict[split] = ds.select(keep_indices)
            filtered_indices_per_split[split] = keep_indices
        return DatasetDict(result_dict), stats_dict, filtered_out_indices, filtered_indices_per_split
    else:
        raise NotImplementedError()
        # For single Dataset (optionally add tqdm here too)
        data = annotated_ds
        keep_indices = []
        filter_stats = defaultdict(lambda: defaultdict(lambda: {
            "count": 0, "tokens": 0, "examples": []
        }))
        filtered_out_indices = []
        for i, ex in enumerate(tqdm(data, desc="Filtering (no split)", total=len(data))):
            filters_trig, ql_type, matched_substring, keyword_counts = filter_and_annotate(ex)
            ex_tokens = len(tokenizer.encode(ex["text"]))
            if not filters_trig:
                keep_indices.append(i)
            else:
                lang = ql_type or "unknown"
                filtered_out_indices.append({
                    "idx": i,
                    "tokens": ex_tokens,
                    "filters": filters_trig,
                    "lang": lang,
                })
                for fil in filters_trig:
                    fs = filter_stats[fil][lang]
                    fs["count"] += 1
                    fs["tokens"] += ex_tokens
                    if "keyword_counts" not in fs:
                        fs["keyword_counts"] = {k: 0 for k in query_keywords}
                    for k in query_keywords:
                        fs["keyword_counts"][k] += keyword_counts.get(k, 0)
                    if len(fs["examples"]) < example_cap:
                        fs["examples"].append({
                            "text": ex["text"],
                            "matched": matched_substring
                        })
        return data.select(keep_indices), filter_stats, filtered_out_indices, keep_indices
    

def annotate_dataset_parallel(dataset, tokenizer, num_proc=NUM_PROC):
    return dataset.map(
        lambda batch: filter_and_annotate(batch, tokenizer=tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
        keep_in_memory=KEEP_IN_MEMORY,
        desc="Annotate & count tokens"
    )


def annotate_datasetdict_parallel(dataset_dict, tokenizer, num_proc=NUM_PROC):
    return DatasetDict({
        k: annotate_dataset_parallel(v, tokenizer, num_proc=num_proc)
        for k, v in dataset_dict.items()
    })

def keep_only_text(example):
    return {"text": example["text"]}

def tokenize_dataset(dataset, tokenizer, num_proc=NUM_PROC, with_indices=WITH_INDICES):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    # Dynamically pick the first split name
    if isinstance(dataset, dict):  # DatasetDict
        split_name = next(iter(dataset.keys()))
        columns = dataset[split_name].column_names
    else:  # Single Dataset
        columns = dataset.column_names
    return dataset.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=columns, with_indices=with_indices,
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
        keep_in_memory=KEEP_IN_MEMORY,)

def group_texts(tokenized_dataset, block_size=512, num_proc=NUM_PROC, with_indices=WITH_INDICES):
    def group_texts_fn(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        # Drop the remainder to have a fixed length (you may also pad if desired)
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

def count_tokens(dataset: DatasetDict | Dataset, tokenizer, num_proc=NUM_PROC, batch_size=1000, show_progress=True):
    """
    Count tokens in a HuggingFace Dataset or DatasetDict in parallel.
    Returns: (total_tokens, total_examples)
    """
    # Handle DatasetDict or plain Dataset
    if isinstance(dataset, dict):  # DatasetDict
        total_tokens, total_examples = 0, 0
        for split in dataset:
            tokens, examples = count_tokens(dataset[split], tokenizer, num_proc=num_proc, batch_size=batch_size, show_progress=show_progress)
            total_tokens += tokens
            total_examples += examples
        return total_tokens, total_examples

    # Fast path for HuggingFace Dataset
    def batch_token_len(batch):
        # tokenizer(batch["text"], ...) returns a dict with "input_ids"
        tok = tokenizer(batch["text"], add_special_tokens=False, truncation=False)
        # If tokenizer returns a list of lists
        if isinstance(tok["input_ids"][0], list):
            lengths = [len(ids) for ids in tok["input_ids"]]
        else:  # If single string
            lengths = [len(tok["input_ids"])]
        return {"num_tokens": lengths}

    # .map is parallel and batched
    mapped = dataset.map(
        batch_token_len,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=[],
        load_from_cache_file=LOAD_FROM_CACHE_FILE,
        keep_in_memory=KEEP_IN_MEMORY,
        desc=f"Counting tokens."
    )
    total_tokens = sum(mapped["num_tokens"])
    total_examples = len(mapped)
    return total_tokens, total_examples



def build_dirty_matched_set(dataset, filtered_out_indices, clean_token_target, filtered_indices_per_split, seed=42, tokenizer=None):
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
                # Count tokens not characters for supplementing
                ex_len = len(tokenizer.encode(data[i]["text"])) if tokenizer else len(data[i]["text"])
                supplement_indices.append(i)
                supplement_tokens += ex_len
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


def collect_stats_for_set(dataset_dict, dataset_type, ref_tokens_per_split, ref_docs_per_split, tokenizer, supplement_stats=None, save=True, dataset_cache_path=None, max_seq_length=512):
    if dataset_cache_path is None:
        raise NotImplementedError("You must provide dataset_cache_path.")
    rows = []
    grouped = {}
    for split in dataset_dict.keys():
        n_docs = len(dataset_dict[split])
        n_tokens, _ = count_tokens(dataset_dict[split], tokenizer)
        # Get number of group blocks
        tknz = tokenize_dataset(DatasetDict({split: dataset_dict[split]}), tokenizer)
        grouped[split] = group_texts(tknz, block_size=max_seq_length)[split]
        n_blocks = len(grouped[split])
        if save and dataset_cache_path:
            os.makedirs(os.path.join(dataset_cache_path, dataset_type, "grouped"), exist_ok=True)
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

    print(f"Grouped splits to save for {dataset_type}:", grouped.keys())
    save_dir = dataset_cache_path
    save_dir = os.path.join(dataset_cache_path, dataset_type, "grouped")
    print(save_dir)
    if save and dataset_cache_path:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving grouped dataset for '{dataset_type}' with splits: {list(grouped.keys())} at {save_dir}")
        if grouped:
            save_dataset(DatasetDict(grouped), save_dir)
        else:
            print(f"Warning: Nothing to save for {dataset_type} at {save_dir} (grouped is empty!)")
    
    return rows