import argparse
import csv
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import (
    AMBIGUOUS_QUERY_KEYWORDS,
    detect_filter_signals,
    detect_strict_query_types,
    load_config,
    query_keywords,
)


class ReservoirSampler:
    def __init__(self, k, seed=42):
        self.k = k
        self.n_seen = 0
        self.items = []
        self.rng = random.Random(seed)

    def add(self, item):
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
            return
        j = self.rng.randrange(self.n_seen)
        if j < self.k:
            self.items[j] = item


class RunningStats:
    def __init__(self):
        self.n = 0
        self.total = 0
        self.min_val = None
        self.max_val = None

    def add(self, x):
        self.n += 1
        self.total += x
        if self.min_val is None or x < self.min_val:
            self.min_val = x
        if self.max_val is None or x > self.max_val:
            self.max_val = x

    def as_dict(self):
        return {
            "n": self.n,
            "total": int(self.total),
            "mean": float(self.total / self.n) if self.n else 0.0,
            "min": int(self.min_val) if self.min_val is not None else 0,
            "max": int(self.max_val) if self.max_val is not None else 0,
        }


def pct(numerator, denominator):
    if not denominator:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def compact_text(text, max_chars=400):
    text = text.replace("\n", " ").replace("\t", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def sample_payload(index, text, signals, strict_types):
    return {
        "index": int(index),
        "filters_triggered": list(signals["filters_triggered"]),
        "filter_ql_type": signals["ql_type"],
        "keyword_counts": dict(signals["keyword_counts"]),
        "strict_query_types": list(strict_types),
        "text_preview": compact_text(text),
    }


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def top_counter(counter, limit=20):
    return [{"name": k, "count": int(v)} for k, v in counter.most_common(limit)]


def to_compact_str(value):
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value)


def analyze_raw_train(raw_train, batch_size, max_samples, sample_size, seed):
    n_total = len(raw_train) if max_samples is None else min(len(raw_train), max_samples)
    totals = Counter()
    trigger_breakdown = Counter()
    filter_ql_types = Counter()
    strict_in_dirty = Counter()
    strict_in_clean = Counter()
    keyword_docfreq_dirty = Counter()
    keyword_occ_dirty = Counter()
    keyword_docfreq_keyword_only = Counter()
    keyword_occ_keyword_only = Counter()
    keyword_combo_dirty = Counter()
    keyword_combo_keyword_only = Counter()
    char_stats_clean = RunningStats()
    char_stats_dirty = RunningStats()

    samplers = {
        "dirty_regex": ReservoirSampler(sample_size, seed + 11),
        "dirty_keyword_only": ReservoirSampler(sample_size, seed + 23),
        "dirty_keyword_only_ambiguous": ReservoirSampler(sample_size, seed + 37),
        "dirty_keyword_only_no_strict": ReservoirSampler(sample_size, seed + 51),
        "clean_strict_leakage": ReservoirSampler(sample_size, seed + 67),
    }

    for start in tqdm(range(0, n_total, batch_size), desc="Raw-train filtering audit", ncols=100):
        end = min(start + batch_size, n_total)
        texts = raw_train[start:end]["text"]
        for offset, text in enumerate(texts):
            idx = start + offset
            signals = detect_filter_signals(text)
            strict_types = detect_strict_query_types(text)

            has_keyword = "keyword" in signals["filters_triggered"]
            has_regex = "regex" in signals["filters_triggered"]
            is_dirty = has_keyword or has_regex
            keywords_present = sorted(signals["keyword_counts"].keys())
            key_combo = "+".join(keywords_present) if keywords_present else "(none)"
            has_strict_query = bool(strict_types)

            totals["total"] += 1

            if is_dirty:
                totals["dirty"] += 1
                char_stats_dirty.add(len(text))

                if has_keyword and has_regex:
                    trigger_breakdown["keyword+regex"] += 1
                elif has_keyword:
                    trigger_breakdown["keyword_only"] += 1
                else:
                    trigger_breakdown["regex_only"] += 1

                if has_regex:
                    totals["dirty_regex"] += 1
                    filter_ql_types[signals["ql_type"] or "unknown"] += 1
                    samplers["dirty_regex"].add(sample_payload(idx, text, signals, strict_types))
                if has_keyword:
                    totals["dirty_keyword"] += 1
                    keyword_combo_dirty[key_combo] += 1
                    for kw, occ in signals["keyword_counts"].items():
                        keyword_docfreq_dirty[kw] += 1
                        keyword_occ_dirty[kw] += int(occ)

                if has_keyword and not has_regex:
                    totals["dirty_keyword_only"] += 1
                    keyword_combo_keyword_only[key_combo] += 1
                    samplers["dirty_keyword_only"].add(sample_payload(idx, text, signals, strict_types))
                    for kw, occ in signals["keyword_counts"].items():
                        keyword_docfreq_keyword_only[kw] += 1
                        keyword_occ_keyword_only[kw] += int(occ)

                    if set(keywords_present).issubset(AMBIGUOUS_QUERY_KEYWORDS):
                        totals["dirty_keyword_only_ambiguous"] += 1
                        samplers["dirty_keyword_only_ambiguous"].add(
                            sample_payload(idx, text, signals, strict_types)
                        )

                if has_strict_query:
                    totals["dirty_strict_query_docs"] += 1
                    for lang in strict_types:
                        strict_in_dirty[lang] += 1
                else:
                    totals["dirty_no_strict_query_docs"] += 1
                    if has_keyword and not has_regex:
                        samplers["dirty_keyword_only_no_strict"].add(
                            sample_payload(idx, text, signals, strict_types)
                        )
            else:
                totals["clean"] += 1
                char_stats_clean.add(len(text))
                trigger_breakdown["none"] += 1
                if has_strict_query:
                    totals["clean_strict_query_docs"] += 1
                    for lang in strict_types:
                        strict_in_clean[lang] += 1
                    samplers["clean_strict_leakage"].add(sample_payload(idx, text, signals, strict_types))

    keyword_rows = []
    for kw in query_keywords:
        keyword_rows.append(
            {
                "keyword": kw,
                "dirty_doc_count": int(keyword_docfreq_dirty.get(kw, 0)),
                "dirty_doc_rate_pct": pct(keyword_docfreq_dirty.get(kw, 0), totals["dirty"]),
                "dirty_occurrences": int(keyword_occ_dirty.get(kw, 0)),
                "keyword_only_doc_count": int(keyword_docfreq_keyword_only.get(kw, 0)),
                "keyword_only_doc_rate_pct": pct(keyword_docfreq_keyword_only.get(kw, 0), totals["dirty_keyword_only"]),
                "keyword_only_occurrences": int(keyword_occ_keyword_only.get(kw, 0)),
            }
        )
    keyword_rows.sort(key=lambda r: (-r["dirty_doc_count"], r["keyword"]))

    combo_rows_dirty = [
        {"keyword_combo": combo, "dirty_doc_count": int(count)}
        for combo, count in keyword_combo_dirty.most_common()
    ]
    combo_rows_keyword_only = [
        {"keyword_combo": combo, "dirty_doc_count": int(count)}
        for combo, count in keyword_combo_keyword_only.most_common()
    ]

    summary = {
        "raw_train": {
            "n_total": int(totals["total"]),
            "n_clean": int(totals["clean"]),
            "n_dirty": int(totals["dirty"]),
            "clean_rate_pct": pct(totals["clean"], totals["total"]),
            "dirty_rate_pct": pct(totals["dirty"], totals["total"]),
        },
        "filter_trigger_breakdown": {
            "keyword_only": {
                "n": int(totals["dirty_keyword_only"]),
                "pct_of_total": pct(totals["dirty_keyword_only"], totals["total"]),
                "pct_of_dirty": pct(totals["dirty_keyword_only"], totals["dirty"]),
            },
            "regex_only": {
                "n": int(trigger_breakdown["regex_only"]),
                "pct_of_total": pct(trigger_breakdown["regex_only"], totals["total"]),
                "pct_of_dirty": pct(trigger_breakdown["regex_only"], totals["dirty"]),
            },
            "keyword_plus_regex": {
                "n": int(trigger_breakdown["keyword+regex"]),
                "pct_of_total": pct(trigger_breakdown["keyword+regex"], totals["total"]),
                "pct_of_dirty": pct(trigger_breakdown["keyword+regex"], totals["dirty"]),
            },
        },
        "query_audit_strict": {
            "dirty_query_docs": {
                "n": int(totals["dirty_strict_query_docs"]),
                "pct_of_dirty": pct(totals["dirty_strict_query_docs"], totals["dirty"]),
            },
            "dirty_non_query_docs": {
                "n": int(totals["dirty_no_strict_query_docs"]),
                "pct_of_dirty": pct(totals["dirty_no_strict_query_docs"], totals["dirty"]),
            },
            "clean_query_leakage_docs": {
                "n": int(totals["clean_strict_query_docs"]),
                "pct_of_clean": pct(totals["clean_strict_query_docs"], totals["clean"]),
            },
            "strict_query_type_counts_in_dirty": top_counter(strict_in_dirty, limit=10),
            "strict_query_type_counts_in_clean": top_counter(strict_in_clean, limit=10),
        },
        "filter_regex_ql_type_counts": top_counter(filter_ql_types, limit=10),
        "keyword_only_ambiguity_proxy": {
            "n_keyword_only_ambiguous": int(totals["dirty_keyword_only_ambiguous"]),
            "pct_of_keyword_only_dirty": pct(totals["dirty_keyword_only_ambiguous"], totals["dirty_keyword_only"]),
            "pct_of_dirty": pct(totals["dirty_keyword_only_ambiguous"], totals["dirty"]),
        },
        "text_length_chars": {
            "clean": char_stats_clean.as_dict(),
            "dirty": char_stats_dirty.as_dict(),
        },
        "top_keyword_combos_dirty": combo_rows_dirty[:25],
        "top_keyword_combos_keyword_only": combo_rows_keyword_only[:25],
    }

    return summary, keyword_rows, samplers


def analyze_streaming_texts(stream, max_samples, sample_size, seed):
    totals = Counter()
    trigger_breakdown = Counter()
    filter_ql_types = Counter()
    strict_in_dirty = Counter()
    strict_in_clean = Counter()
    keyword_docfreq_dirty = Counter()
    keyword_occ_dirty = Counter()
    keyword_docfreq_keyword_only = Counter()
    keyword_occ_keyword_only = Counter()
    keyword_combo_dirty = Counter()
    keyword_combo_keyword_only = Counter()
    char_stats_clean = RunningStats()
    char_stats_dirty = RunningStats()

    samplers = {
        "dirty_regex": ReservoirSampler(sample_size, seed + 11),
        "dirty_keyword_only": ReservoirSampler(sample_size, seed + 23),
        "dirty_keyword_only_ambiguous": ReservoirSampler(sample_size, seed + 37),
        "dirty_keyword_only_no_strict": ReservoirSampler(sample_size, seed + 51),
        "clean_strict_leakage": ReservoirSampler(sample_size, seed + 67),
    }

    pbar = tqdm(
        total=max_samples if max_samples is not None else None,
        desc="Streaming filtering audit",
        ncols=100,
    )
    for idx, sample in enumerate(stream):
        if max_samples is not None and idx >= max_samples:
            break

        text = sample["text"]
        signals = detect_filter_signals(text)
        strict_types = detect_strict_query_types(text)

        has_keyword = "keyword" in signals["filters_triggered"]
        has_regex = "regex" in signals["filters_triggered"]
        is_dirty = has_keyword or has_regex
        keywords_present = sorted(signals["keyword_counts"].keys())
        key_combo = "+".join(keywords_present) if keywords_present else "(none)"
        has_strict_query = bool(strict_types)

        totals["total"] += 1

        if is_dirty:
            totals["dirty"] += 1
            char_stats_dirty.add(len(text))

            if has_keyword and has_regex:
                trigger_breakdown["keyword+regex"] += 1
            elif has_keyword:
                trigger_breakdown["keyword_only"] += 1
            else:
                trigger_breakdown["regex_only"] += 1

            if has_regex:
                totals["dirty_regex"] += 1
                filter_ql_types[signals["ql_type"] or "unknown"] += 1
                samplers["dirty_regex"].add(sample_payload(idx, text, signals, strict_types))
            if has_keyword:
                totals["dirty_keyword"] += 1
                keyword_combo_dirty[key_combo] += 1
                for kw, occ in signals["keyword_counts"].items():
                    keyword_docfreq_dirty[kw] += 1
                    keyword_occ_dirty[kw] += int(occ)

            if has_keyword and not has_regex:
                totals["dirty_keyword_only"] += 1
                keyword_combo_keyword_only[key_combo] += 1
                samplers["dirty_keyword_only"].add(sample_payload(idx, text, signals, strict_types))
                for kw, occ in signals["keyword_counts"].items():
                    keyword_docfreq_keyword_only[kw] += 1
                    keyword_occ_keyword_only[kw] += int(occ)

                if set(keywords_present).issubset(AMBIGUOUS_QUERY_KEYWORDS):
                    totals["dirty_keyword_only_ambiguous"] += 1
                    samplers["dirty_keyword_only_ambiguous"].add(
                        sample_payload(idx, text, signals, strict_types)
                    )

            if has_strict_query:
                totals["dirty_strict_query_docs"] += 1
                for lang in strict_types:
                    strict_in_dirty[lang] += 1
            else:
                totals["dirty_no_strict_query_docs"] += 1
                if has_keyword and not has_regex:
                    samplers["dirty_keyword_only_no_strict"].add(
                        sample_payload(idx, text, signals, strict_types)
                    )
        else:
            totals["clean"] += 1
            char_stats_clean.add(len(text))
            trigger_breakdown["none"] += 1
            if has_strict_query:
                totals["clean_strict_query_docs"] += 1
                for lang in strict_types:
                    strict_in_clean[lang] += 1
                samplers["clean_strict_leakage"].add(sample_payload(idx, text, signals, strict_types))

        pbar.update(1)
    pbar.close()

    keyword_rows = []
    for kw in query_keywords:
        keyword_rows.append(
            {
                "keyword": kw,
                "dirty_doc_count": int(keyword_docfreq_dirty.get(kw, 0)),
                "dirty_doc_rate_pct": pct(keyword_docfreq_dirty.get(kw, 0), totals["dirty"]),
                "dirty_occurrences": int(keyword_occ_dirty.get(kw, 0)),
                "keyword_only_doc_count": int(keyword_docfreq_keyword_only.get(kw, 0)),
                "keyword_only_doc_rate_pct": pct(keyword_docfreq_keyword_only.get(kw, 0), totals["dirty_keyword_only"]),
                "keyword_only_occurrences": int(keyword_occ_keyword_only.get(kw, 0)),
            }
        )
    keyword_rows.sort(key=lambda r: (-r["dirty_doc_count"], r["keyword"]))

    combo_rows_dirty = [
        {"keyword_combo": combo, "dirty_doc_count": int(count)}
        for combo, count in keyword_combo_dirty.most_common()
    ]
    combo_rows_keyword_only = [
        {"keyword_combo": combo, "dirty_doc_count": int(count)}
        for combo, count in keyword_combo_keyword_only.most_common()
    ]

    summary = {
        "raw_train": {
            "n_total": int(totals["total"]),
            "n_clean": int(totals["clean"]),
            "n_dirty": int(totals["dirty"]),
            "clean_rate_pct": pct(totals["clean"], totals["total"]),
            "dirty_rate_pct": pct(totals["dirty"], totals["total"]),
        },
        "filter_trigger_breakdown": {
            "keyword_only": {
                "n": int(totals["dirty_keyword_only"]),
                "pct_of_total": pct(totals["dirty_keyword_only"], totals["total"]),
                "pct_of_dirty": pct(totals["dirty_keyword_only"], totals["dirty"]),
            },
            "regex_only": {
                "n": int(trigger_breakdown["regex_only"]),
                "pct_of_total": pct(trigger_breakdown["regex_only"], totals["total"]),
                "pct_of_dirty": pct(trigger_breakdown["regex_only"], totals["dirty"]),
            },
            "keyword_plus_regex": {
                "n": int(trigger_breakdown["keyword+regex"]),
                "pct_of_total": pct(trigger_breakdown["keyword+regex"], totals["total"]),
                "pct_of_dirty": pct(trigger_breakdown["keyword+regex"], totals["dirty"]),
            },
        },
        "query_audit_strict": {
            "dirty_query_docs": {
                "n": int(totals["dirty_strict_query_docs"]),
                "pct_of_dirty": pct(totals["dirty_strict_query_docs"], totals["dirty"]),
            },
            "dirty_non_query_docs": {
                "n": int(totals["dirty_no_strict_query_docs"]),
                "pct_of_dirty": pct(totals["dirty_no_strict_query_docs"], totals["dirty"]),
            },
            "clean_query_leakage_docs": {
                "n": int(totals["clean_strict_query_docs"]),
                "pct_of_clean": pct(totals["clean_strict_query_docs"], totals["clean"]),
            },
            "strict_query_type_counts_in_dirty": top_counter(strict_in_dirty, limit=10),
            "strict_query_type_counts_in_clean": top_counter(strict_in_clean, limit=10),
        },
        "filter_regex_ql_type_counts": top_counter(filter_ql_types, limit=10),
        "keyword_only_ambiguity_proxy": {
            "n_keyword_only_ambiguous": int(totals["dirty_keyword_only_ambiguous"]),
            "pct_of_keyword_only_dirty": pct(totals["dirty_keyword_only_ambiguous"], totals["dirty_keyword_only"]),
            "pct_of_dirty": pct(totals["dirty_keyword_only_ambiguous"], totals["dirty"]),
        },
        "text_length_chars": {
            "clean": char_stats_clean.as_dict(),
            "dirty": char_stats_dirty.as_dict(),
        },
        "top_keyword_combos_dirty": combo_rows_dirty[:25],
        "top_keyword_combos_keyword_only": combo_rows_keyword_only[:25],
    }

    return summary, keyword_rows, samplers


def audit_grouped_blocks(grouped_ds, tokenizer, strict_batch_size, sample_size, seed, label):
    total_blocks = len(grouped_ds)
    query_blocks = 0
    strict_type_counts = Counter()
    samples = ReservoirSampler(sample_size, seed)

    for start in tqdm(range(0, total_blocks, strict_batch_size), desc=f"Grouped strict audit: {label}", ncols=100):
        end = min(start + strict_batch_size, total_blocks)
        batch = grouped_ds[start:end]
        decoded = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        for offset, text in enumerate(decoded):
            strict_types = detect_strict_query_types(text)
            if strict_types:
                idx = start + offset
                query_blocks += 1
                for lang in strict_types:
                    strict_type_counts[lang] += 1
                samples.add(
                    {
                        "index": int(idx),
                        "strict_query_types": list(strict_types),
                        "text_preview": compact_text(text),
                    }
                )

    return {
        "n_blocks_total": int(total_blocks),
        "n_blocks_with_strict_query": int(query_blocks),
        "pct_blocks_with_strict_query": pct(query_blocks, total_blocks),
        "strict_query_type_counts": top_counter(strict_type_counts, limit=10),
        "sample_records": samples.items,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed filtering/removed-sample analysis for pretraining data.")
    parser.add_argument(
        "ds",
        choices=(
            "openwebtext",
            "openwebtext-10k",
            "openwebtext-dirty",
            "openwebtext-injected",
            "realnewslike",
            "c4-dirty",
        ),
    )
    parser.add_argument("--dataset_cache_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--strict_batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit_grouped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode and audit final grouped datasets for strict SQL/SPARQL/Cypher leakage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_args = load_config(args.ds)
    dataset_cache_path = args.dataset_cache_path or config_args.get(
        "dataset_cache_path", f"./{args.ds}-preproc-chunks"
    )
    output_dir = args.output_dir or os.path.join("stats", f"filtering_analysis_{args.ds}")
    os.makedirs(output_dir, exist_ok=True)

    raw_train_path = os.path.join(dataset_cache_path, "raw_train")
    if os.path.exists(raw_train_path):
        print(f"Loading raw_train from: {raw_train_path}")
        raw_train = load_from_disk(raw_train_path)
        raw_summary, keyword_rows, samplers = analyze_raw_train(
            raw_train=raw_train,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    else:
        dataset_name = config_args.get("dataset")
        subdataset = config_args.get("subdataset")
        cache_dir = config_args.get("cache_dir")
        if not dataset_name:
            raise FileNotFoundError(
                f"raw_train dataset not found at {raw_train_path}, and config does not define 'dataset' for streaming fallback."
            )
        print(
            f"raw_train not found at {raw_train_path}. "
            f"Falling back to streaming source {dataset_name}/{subdataset or ''} split=train."
        )
        stream = load_dataset(
            dataset_name,
            subdataset,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        raw_summary, keyword_rows, samplers = analyze_streaming_texts(
            stream=stream,
            max_samples=args.max_samples,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    grouped_summary = None
    grouped_clean_samples = []
    grouped_dirty_samples = []
    if args.audit_grouped:
        model_name = config_args.get("model_name_or_path")
        clean_path = os.path.join(dataset_cache_path, "final_clean_grouped")
        dirty_path = os.path.join(dataset_cache_path, "final_dirtymatched_grouped")
        if model_name and os.path.exists(clean_path) and os.path.exists(dirty_path):
            try:
                print(f"Loading tokenizer: {model_name}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"Loading grouped clean from: {clean_path}")
                grouped_clean = load_from_disk(clean_path)
                print(f"Loading grouped dirtymatched from: {dirty_path}")
                grouped_dirty = load_from_disk(dirty_path)
                grouped_clean_summary = audit_grouped_blocks(
                    grouped_ds=grouped_clean,
                    tokenizer=tokenizer,
                    strict_batch_size=args.strict_batch_size,
                    sample_size=args.sample_size,
                    seed=args.seed + 101,
                    label="clean",
                )
                grouped_dirty_summary = audit_grouped_blocks(
                    grouped_ds=grouped_dirty,
                    tokenizer=tokenizer,
                    strict_batch_size=args.strict_batch_size,
                    sample_size=args.sample_size,
                    seed=args.seed + 131,
                    label="dirtymatched",
                )
                grouped_clean_samples = grouped_clean_summary.pop("sample_records")
                grouped_dirty_samples = grouped_dirty_summary.pop("sample_records")
                grouped_summary = {
                    "final_clean_grouped": grouped_clean_summary,
                    "final_dirtymatched_grouped": grouped_dirty_summary,
                }
            except Exception as e:
                print(f"Skipping grouped audit due to tokenizer/decode failure: {e}")
        else:
            print("Skipping grouped audit: missing tokenizer config or grouped dataset paths.")

    summary = {
        "run_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.ds,
            "dataset_cache_path": dataset_cache_path,
            "raw_train_path": raw_train_path,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "strict_batch_size": args.strict_batch_size,
            "sample_size": args.sample_size,
            "audit_grouped": args.audit_grouped,
        },
        "raw_filtering_audit": raw_summary,
    }
    if grouped_summary is not None:
        summary["grouped_dataset_audit"] = grouped_summary

    summary_path = os.path.join(output_dir, "summary.json")
    write_json(summary_path, summary)
    write_csv(
        os.path.join(output_dir, "keyword_trigger_stats.csv"),
        keyword_rows,
        fieldnames=[
            "keyword",
            "dirty_doc_count",
            "dirty_doc_rate_pct",
            "dirty_occurrences",
            "keyword_only_doc_count",
            "keyword_only_doc_rate_pct",
            "keyword_only_occurrences",
        ],
    )
    write_jsonl(os.path.join(output_dir, "samples_dirty_regex.jsonl"), samplers["dirty_regex"].items)
    write_jsonl(
        os.path.join(output_dir, "samples_dirty_keyword_only.jsonl"),
        samplers["dirty_keyword_only"].items,
    )
    write_jsonl(
        os.path.join(output_dir, "samples_dirty_keyword_only_ambiguous.jsonl"),
        samplers["dirty_keyword_only_ambiguous"].items,
    )
    write_jsonl(
        os.path.join(output_dir, "samples_dirty_keyword_only_no_strict.jsonl"),
        samplers["dirty_keyword_only_no_strict"].items,
    )
    write_jsonl(
        os.path.join(output_dir, "samples_clean_strict_leakage.jsonl"),
        samplers["clean_strict_leakage"].items,
    )
    manual_rows = []
    category_to_records = {
        "dirty_regex": samplers["dirty_regex"].items,
        "dirty_keyword_only": samplers["dirty_keyword_only"].items,
        "dirty_keyword_only_ambiguous": samplers["dirty_keyword_only_ambiguous"].items,
        "dirty_keyword_only_no_strict": samplers["dirty_keyword_only_no_strict"].items,
        "clean_strict_leakage": samplers["clean_strict_leakage"].items,
    }
    for category, records in category_to_records.items():
        for rec in records:
            manual_rows.append(
                {
                    "category": category,
                    "index": rec.get("index", -1),
                    "filters_triggered": to_compact_str(rec.get("filters_triggered", [])),
                    "filter_ql_type": rec.get("filter_ql_type", ""),
                    "keyword_counts": to_compact_str(rec.get("keyword_counts", {})),
                    "strict_query_types": to_compact_str(rec.get("strict_query_types", [])),
                    "text_preview": rec.get("text_preview", ""),
                    "manual_label": "",
                    "notes": "",
                }
            )
    write_csv(
        os.path.join(output_dir, "manual_review_samples.csv"),
        manual_rows,
        fieldnames=[
            "category",
            "index",
            "filters_triggered",
            "filter_ql_type",
            "keyword_counts",
            "strict_query_types",
            "text_preview",
            "manual_label",
            "notes",
        ],
    )
    if grouped_summary is not None:
        write_jsonl(os.path.join(output_dir, "samples_grouped_clean_strict_queries.jsonl"), grouped_clean_samples)
        write_jsonl(os.path.join(output_dir, "samples_grouped_dirty_strict_queries.jsonl"), grouped_dirty_samples)

    n_total = raw_summary["raw_train"]["n_total"]
    n_dirty = raw_summary["raw_train"]["n_dirty"]
    n_clean = raw_summary["raw_train"]["n_clean"]
    n_dirty_strict = raw_summary["query_audit_strict"]["dirty_query_docs"]["n"]
    n_clean_leak = raw_summary["query_audit_strict"]["clean_query_leakage_docs"]["n"]

    print("\n=== Filtering Analysis Summary ===")
    print(f"Total raw_train samples analyzed: {n_total}")
    print(f"Dirty (filtered out): {n_dirty} ({raw_summary['raw_train']['dirty_rate_pct']}%)")
    print(f"Clean (kept): {n_clean} ({raw_summary['raw_train']['clean_rate_pct']}%)")
    print(
        f"Dirty strict-query docs (SQL/SPARQL/Cypher): {n_dirty_strict} "
        f"({raw_summary['query_audit_strict']['dirty_query_docs']['pct_of_dirty']}% of dirty)"
    )
    print(
        f"Clean strict-query leakage docs: {n_clean_leak} "
        f"({raw_summary['query_audit_strict']['clean_query_leakage_docs']['pct_of_clean']}% of clean)"
    )
    print(f"Saved summary to: {summary_path}")
    print(f"Saved keyword stats to: {os.path.join(output_dir, 'keyword_trigger_stats.csv')}")
    print(f"Saved sampled records to: {output_dir}")


if __name__ == "__main__":
    main()
