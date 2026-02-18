import argparse
import csv
import json
import os
import re
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from data_utils import detect_filter_signals, detect_strict_query_types, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hydrate sampled rows with full raw_train text using stored dataset indexes."
    )
    parser.add_argument("--input_csv", required=True, type=str, help="Path to manual_review_samples.csv")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <input>_full.csv",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional output JSONL path. Defaults to <input>_full.jsonl",
    )
    parser.add_argument(
        "--raw_train_path",
        type=str,
        default=None,
        help="Explicit raw_train dataset path. If omitted, inferred from summary.json or --ds config.",
    )
    parser.add_argument(
        "--ds",
        choices=("openwebtext", "openwebtext-10k", "realnewslike"),
        default=None,
        help="Dataset config key fallback for raw_train_path inference.",
    )
    parser.add_argument(
        "--snippet_window",
        type=int,
        default=160,
        help="Context characters before and after first trigger match.",
    )
    return parser.parse_args()


def infer_raw_train_path(input_csv, raw_train_path=None, ds=None):
    if raw_train_path:
        return raw_train_path

    summary_path = os.path.join(os.path.dirname(input_csv), "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        meta = summary.get("run_meta", {})
        if meta.get("raw_train_path"):
            return meta["raw_train_path"]
        if meta.get("dataset_cache_path"):
            return os.path.join(meta["dataset_cache_path"], "raw_train")

    if ds:
        cfg = load_config(ds)
        cache_path = cfg.get("dataset_cache_path")
        if cache_path:
            return os.path.join(cache_path, "raw_train")

    raise ValueError(
        "Could not infer raw_train path. Provide --raw_train_path explicitly, or run from a stats dir with summary.json, or pass --ds."
    )


def parse_keyword_counts(raw_value):
    if not raw_value:
        return {}
    try:
        value = json.loads(raw_value)
        if isinstance(value, dict):
            return value
        return {}
    except Exception:
        return {}


def find_trigger_context(text, filters_triggered, keyword_counts, snippet_window):
    filters = set((filters_triggered or "").split("|")) if isinstance(filters_triggered, str) else set()

    if "regex" in filters:
        signals = detect_filter_signals(text)
        match_text = signals.get("matched_substring", "")
        if match_text:
            start_idx = text.lower().find(match_text.lower())
            if start_idx >= 0:
                end_idx = start_idx + len(match_text)
                left = max(0, start_idx - snippet_window)
                right = min(len(text), end_idx + snippet_window)
                return {
                    "trigger_kind": "regex",
                    "trigger_text": match_text,
                    "trigger_start": int(start_idx),
                    "trigger_end": int(end_idx),
                    "trigger_context": text[left:right],
                }

    for kw in keyword_counts.keys():
        hit = re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)
        if hit:
            start_idx, end_idx = hit.start(), hit.end()
            left = max(0, start_idx - snippet_window)
            right = min(len(text), end_idx + snippet_window)
            return {
                "trigger_kind": "keyword",
                "trigger_text": text[start_idx:end_idx],
                "trigger_start": int(start_idx),
                "trigger_end": int(end_idx),
                "trigger_context": text[left:right],
            }

    return {
        "trigger_kind": "",
        "trigger_text": "",
        "trigger_start": -1,
        "trigger_end": -1,
        "trigger_context": "",
    }


def main():
    args = parse_args()
    input_csv = args.input_csv
    base = os.path.splitext(input_csv)[0]
    output_csv = args.output_csv or f"{base}_full.csv"
    output_jsonl = args.output_jsonl or f"{base}_full.jsonl"

    raw_train_path = infer_raw_train_path(input_csv, raw_train_path=args.raw_train_path, ds=args.ds)
    if not os.path.exists(raw_train_path):
        raise FileNotFoundError(f"raw_train dataset not found at: {raw_train_path}")

    print(f"Loading samples from: {input_csv}")
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    print(f"Loading raw_train from: {raw_train_path}")
    raw_train = load_from_disk(raw_train_path)
    ds_len = len(raw_train)

    hydrated_rows = []
    for row in tqdm(rows, desc="Hydrating rows by index", ncols=100):
        idx = int(row.get("index", -1))
        if idx < 0 or idx >= ds_len:
            hydrated = dict(row)
            hydrated["hydrate_error"] = f"index_out_of_range({idx})"
            hydrated_rows.append(hydrated)
            continue

        text = raw_train[idx]["text"]
        keyword_counts = parse_keyword_counts(row.get("keyword_counts", ""))
        trigger = find_trigger_context(
            text=text,
            filters_triggered=row.get("filters_triggered", ""),
            keyword_counts=keyword_counts,
            snippet_window=args.snippet_window,
        )
        strict_types = detect_strict_query_types(text)

        hydrated = dict(row)
        hydrated["hydrate_error"] = ""
        hydrated["text_char_len"] = str(len(text))
        hydrated["strict_query_types_recomputed"] = "|".join(strict_types)
        hydrated["trigger_kind"] = trigger["trigger_kind"]
        hydrated["trigger_text"] = trigger["trigger_text"]
        hydrated["trigger_start"] = str(trigger["trigger_start"])
        hydrated["trigger_end"] = str(trigger["trigger_end"])
        hydrated["trigger_context"] = trigger["trigger_context"].replace("\n", " ").replace("\t", " ").strip()
        hydrated["text_full"] = text
        hydrated_rows.append(hydrated)

    fieldnames = list(hydrated_rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hydrated_rows)

    with open(output_jsonl, "w") as f:
        for row in hydrated_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote hydrated CSV: {output_csv}")
    print(f"Wrote hydrated JSONL: {output_jsonl}")
    print(f"Rows processed: {len(hydrated_rows)}")
    print(f"Dataset size checked: {ds_len}")


if __name__ == "__main__":
    main()
