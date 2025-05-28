import re
from datasets import load_dataset, DatasetDict, Dataset
from itertools import chain
from collections import defaultdict, Counter
from tqdm import tqdm

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

def filter_and_annotate(example):
    filters_triggered = []
    ql_type = None
    matched_substring = None

    tokens_upper = re.findall(r'\w+', example["text"].upper())
    token_counts = Counter(tokens_upper)
    keyword_counts = {kw: token_counts[kw] for kw in query_keywords if token_counts[kw] > 0}
    if keyword_counts:
        filters_triggered.append("keyword")

    # Regex-based filter (capture what was matched)
    for pattern, lang in QL_PATTERNS:
        match = re.search(pattern, example["text"], re.IGNORECASE)
        if match:
            filters_triggered.append("regex")
            ql_type = lang
            matched_substring = match.group(0)
            break

    return filters_triggered, ql_type, matched_substring, keyword_counts


# query_keywords = [
#     "SELECT", "MATCH", "WHERE", "JOIN", "OPTIONAL", "FILTER",
#     "RETURN", "INSERT", "DELETE", "SQL", "SPARQL", "CYPHER"
# ]
# query_keywords_set = set(query_keywords)


# def _has_keywords_in_window(positions: list[int], window_len: int):
#     # positions: sorted list of all keyword positions (ints)
#     left = 0
#     for right in range(1, len(positions)):
#         while positions[right] - positions[left] > window_len:
#             left += 1
#         if right - left + 1 >= 2:
#             return True
#     return False

# def filter_and_annotate(example, window_len=64):
#     filters_triggered = []
#     ql_type = None
#     matched_substring = None

#     # Tokenize and upper-case
#     tokens = re.findall(r'\w+', example["text"].upper())
#     keyword_counts = {k: 0 for k in query_keywords}
#     positions = {k: [] for k in query_keywords}

#     for idx, token in enumerate(tokens):
#         if token in query_keywords_set:
#             keyword_counts[token] += 1
#             positions[token].append(idx)

#     # Check for at least two (could be same or different) within `window_len` tokens:
#     all_keyword_positions = sorted([pos for plist in positions.values() for pos in plist])
#     keyword_in_window = _has_keywords_in_window(all_keyword_positions, window_len)

#     if keyword_in_window:
#         filters_triggered.append("keyword")

#     # Regex-based filter
#     for pattern, lang in QL_PATTERNS:
#         match = re.search(pattern, example["text"], re.IGNORECASE)
#         if match:
#             filters_triggered.append("regex")
#             ql_type = lang
#             matched_substring = match.group(0)
#             break

#     return filters_triggered, ql_type, matched_substring, keyword_counts


def filter_dataset_with_stats(dataset, tokenizer, example_cap=10):
    """
    Filters the dataset and simultaneously collects filter statistics.
    Returns: filtered_dataset (matching original HF DatasetDict), filter_stats dictionary
    """
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        filtered_dict = {}
        filter_stats = defaultdict(lambda: defaultdict(lambda: {
            "count": 0, "tokens": 0, "examples": []
        }))
        filtered_out_indices = defaultdict(list)
        filtered_indices_per_split = {}
        for split in tqdm(splits, desc="Filtering splits"):
            data = dataset[split]
            keep_indices = []
            for i, ex in enumerate(tqdm(data, desc=f"Filtering {split}", leave=False, total=len(data))):
                filters_trig, ql_type, matched_substring, keyword_counts = filter_and_annotate(ex)
                ex_tokens = len(tokenizer.encode(ex["text"]))
                if not filters_trig:
                    keep_indices.append(i)
                else:
                    lang = ql_type or "unknown"
                    filtered_out_indices[split].append({
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
            filtered_dict[split] = data.select(keep_indices)
            filtered_indices_per_split[split] = keep_indices
        return DatasetDict(filtered_dict), filter_stats, filtered_out_indices, filtered_indices_per_split
    else:
        # For single Dataset (optionally add tqdm here too)
        data = dataset
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
    
    
def tokenize_dataset(dataset, tokenizer, num_proc=16):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    # Dynamically pick the first split name
    if isinstance(dataset, dict):  # DatasetDict
        split_name = next(iter(dataset.keys()))
        columns = dataset[split_name].column_names
    else:  # Single Dataset
        columns = dataset.column_names
    return dataset.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=columns)

def group_texts(tokenized_dataset, block_size=512, num_proc=8):
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

    return tokenized_dataset.map(group_texts_fn, batched=True, num_proc=num_proc)

def save_dataset(dataset, path):
    dataset.save_to_disk(path)

def count_tokens(dataset, tokenizer):
    # If dataset is HuggingFace split: dataset["train"]
    # If dataset is a list of dicts: dataset
    if isinstance(dataset, dict) and "train" in dataset:
        data_list = dataset["train"]
    else:
        data_list = dataset
    return sum(len(tokenizer.encode(d["text"])) for d in data_list), len(data_list)