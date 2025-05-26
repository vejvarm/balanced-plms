import re
from datasets import load_dataset, DatasetDict, Dataset
from itertools import chain
from collections import defaultdict

# Patterns for query languages and their names
QL_PATTERNS = [
    (r"\bSELECT\b.*\bFROM\b", "SQL"),
    # (r"\bSELECT\s+.+?\s+FROM\s+.+?(WHERE|GROUP\s+BY|ORDER\s+BY|;|$)", "SQL"),
    (r"\bMATCH\b.*\bRETURN\b", "Cypher"),
    (r"\bselect\b.*\bwhere\b", "SPARQL"),
    # (r"(db\.\w+\.[^;]+)", "MQL"),   # MQL (https://github.com/jf87/SM3-Text-to-Query/blob/main/src/evaluation/DB_query.py)
]

def load_and_split_data(dataset: str, subdataset: str | None, cache_dir, split_fraction="train", test_size=10000) -> Dataset | DatasetDict:
    dataset: Dataset | DatasetDict = load_dataset(dataset, subdataset, split=split_fraction, cache_dir=cache_dir, trust_remote_code=True)
    return dataset.train_test_split(test_size=test_size, writer_batch_size=100000)


def filter_and_annotate(example):
    filters_triggered = []
    ql_type = None
    matched_substring = None

    # Regex-based filter (capture what was matched)
    for pattern, lang in QL_PATTERNS:
        match = re.search(pattern, example["text"], re.IGNORECASE)
        if match:
            filters_triggered.append("regex")
            ql_type = lang
            matched_substring = match.group(0)
            break

    # Keyword filter
    query_keywords = ["SELECT", "MATCH", "WHERE", "JOIN", "OPTIONAL", "FILTER", "RETURN", "INSERT", "DELETE"]
    tokens = re.findall(r'\w+', example["text"].upper())
    for keyword in query_keywords:
        if keyword in tokens:
            filters_triggered.append("keyword")
            break

    return filters_triggered, ql_type, matched_substring

def filter_dataset_with_stats(dataset, tokenizer, example_cap=10):
    """
    Filters the dataset and simultaneously collects filter statistics.
    Returns: filtered_dataset (matching original HF DatasetDict), filter_stats dictionary
    """
    if isinstance(dataset, DatasetDict):
        splits = dataset.keys()
        filtered_dict = {}
        filter_stats = defaultdict(lambda: defaultdict(lambda: {
            "count": 0, "tokens": 0, "examples": []
        }))
        for split in splits:
            data = dataset[split]
            keep_indices = []
            for i, ex in enumerate(data):
                filters_trig, ql_type, matched_substring = filter_and_annotate(ex)
                ex_tokens = len(tokenizer.encode(ex["text"]))
                if not filters_trig:
                    keep_indices.append(i)
                else:
                    lang = ql_type or "unknown"
                    for fil in filters_trig:
                        fs = filter_stats[fil][lang]
                        fs["count"] += 1
                        fs["tokens"] += ex_tokens
                        if len(fs["examples"]) < example_cap:
                            fs["examples"].append({
                                "text": ex["text"],
                                "matched": matched_substring
                            })
            # Subset the data after filtering
            filtered_dict[split] = data.select(keep_indices)
        return DatasetDict(filtered_dict), filter_stats
    else:
        # For single Dataset
        data = dataset
        keep_indices = []
        filter_stats = defaultdict(lambda: defaultdict(lambda: {
            "count": 0, "tokens": 0, "examples": []
        }))
        for i, ex in enumerate(data):
            filters_trig, ql_type = filter_and_annotate(ex)
            ex_tokens = len(tokenizer.encode(ex["text"]))
            if not filters_trig:
                keep_indices.append(i)
            else:
                lang = ql_type or "unknown"
                for fil in filters_trig:
                    fs = filter_stats[fil][lang]
                    fs["count"] += 1
                    fs["tokens"] += ex_tokens
                    if len(fs["examples"]) < example_cap:
                        fs["examples"].append(ex["text"])
        return data.select(keep_indices), filter_stats

def tokenize_dataset(dataset, tokenizer, num_proc=16):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    return dataset.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=dataset["train"].column_names)

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