import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import (
    AMBIGUOUS_QUERY_KEYWORDS,
    detect_filter_signals,
    detect_strict_query_types,
    load_config,
    query_keywords,
)


class RunningStats:
    def __init__(self):
        self.n = 0
        self.total = 0
        self.min_val = None
        self.max_val = None

    def add(self, value):
        self.n += 1
        self.total += value
        if self.min_val is None or value < self.min_val:
            self.min_val = value
        if self.max_val is None or value > self.max_val:
            self.max_val = value

    def to_dict(self):
        return {
            "n": int(self.n),
            "total": int(self.total),
            "mean": float(self.total / self.n) if self.n else 0.0,
            "min": int(self.min_val) if self.min_val is not None else 0,
            "max": int(self.max_val) if self.max_val is not None else 0,
        }


def pct(numerator, denominator):
    if not denominator:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def top_counter(counter, limit=25):
    return [{"name": k, "count": int(v)} for k, v in counter.most_common(limit)]


def compact_text(text, max_chars=400):
    text = text.replace("\n", " ").replace("\t", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def to_compact_str(value):
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value)


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


def get_nested_value(obj, dotted_key):
    curr = obj
    for part in dotted_key.split("."):
        if not isinstance(curr, dict) or part not in curr:
            raise KeyError(f"Could not resolve key '{dotted_key}'. Missing '{part}'.")
        curr = curr[part]
    return curr


def resolve_target_tokens(args):
    if args.target_tokens is not None:
        return int(args.target_tokens)

    if args.target_stats_path:
        stats_path = args.target_stats_path
    else:
        cfg = load_config(args.target_ds)
        cache_path = cfg.get("dataset_cache_path")
        if cache_path is None:
            raise ValueError(f"dataset_cache_path missing in config for target_ds='{args.target_ds}'")
        stats_path = os.path.join(cache_path, "final_stats.json")

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Target stats file not found: {stats_path}")

    with open(stats_path) as f:
        stats = json.load(f)
    return int(get_nested_value(stats, args.target_stats_field))


def resolve_model_name(args):
    if args.model_name_or_path:
        return args.model_name_or_path
    cfg = load_config(args.target_ds)
    model_name = cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError(f"model_name_or_path missing in config for target_ds='{args.target_ds}'")
    return model_name


def load_tokenizer(model_name):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                "Failed to load tokenizer. Provide a local path via --model_name_or_path, "
                "or pre-download tokenizer files to local cache."
            ) from e


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-pass streaming builder: collect clean and dirty grouped datasets with configurable "
            "dirty token ratio, and write filtering audit stats/samples."
        )
    )
    parser.add_argument("--source_dataset", type=str, default="allenai/c4")
    parser.add_argument("--source_config", type=str, default="en.noclean")
    parser.add_argument("--source_split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default="/work/datasets")

    parser.add_argument(
        "--target_ds",
        type=str,
        choices=(
            "openwebtext",
            "openwebtext-10k",
            "openwebtext-dirty",
            "openwebtext-injected",
            "realnewslike",
            "c4-dirty",
        ),
        default="openwebtext",
        help="Config used to infer target tokens and default tokenizer when not explicitly provided.",
    )
    parser.add_argument("--target_tokens", type=int, default=None)
    parser.add_argument("--target_stats_path", type=str, default=None)
    parser.add_argument("--target_stats_field", type=str, default="final_clean.num_tokens")

    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--chunk_blocks", type=int, default=20000)

    parser.add_argument(
        "--dirty_rate",
        type=float,
        default=0.1,
        help="Target dirty token fraction in final output (0.0 - 1.0).",
    )
    parser.add_argument(
        "--dirty_selection_mode",
        type=str,
        choices=("regex", "strict", "regex_or_strict", "keyword_or_regex"),
        default="regex",
        help="How dirty candidates are selected from stream.",
    )
    parser.add_argument(
        "--clean_exclude_strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude strict query matches from clean candidates in addition to keyword+regex filtering.",
    )
    parser.add_argument(
        "--dirty_pool_multiplier",
        type=float,
        default=1.0,
        help=(
            "Collect extra dirty pool blocks in one pass. Final dirtymatched keeps only the target "
            "dirty_rate fraction. Use >1.0 to allow remixing later without re-streaming."
        ),
    )
    parser.add_argument(
        "--clean_pool_multiplier",
        type=float,
        default=1.0,
        help=(
            "Collect extra clean pool blocks in one pass. Use >1.0 if you also want to "
            "support lowering dirty rate later without re-streaming."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work/datasets/c4-en-noclean-match-owt-clean",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max_source_rows", type=int, default=None)
    parser.add_argument("--log_every_rows", type=int, default=10000)
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument(
        "--shared_dev_from",
        type=str,
        default="/work/datasets/openwebtext-preproc/shared_dev_grouped/grouped",
        help="Optional grouped dev dataset path to symlink/copy under output_dir/shared_dev_grouped/grouped.",
    )
    parser.add_argument(
        "--shared_dev_mode",
        choices=("symlink", "copy", "skip"),
        default="symlink",
    )
    return parser.parse_args()


def load_progress(progress_path):
    if not os.path.exists(progress_path):
        return None
    with open(progress_path) as f:
        return json.load(f)


def init_side():
    return {
        "tok_keys": None,
        "carry": {},
        "offset": 0,
        "block_buffer": {},
        "source_rows_used": 0,
        "written_blocks": 0,
        "written_tokens": 0,
        "chunks_written": 0,
    }


def ensure_side_keys(side, tok):
    if side["tok_keys"] is None:
        side["tok_keys"] = sorted(tok.keys())
        side["carry"] = {k: [] for k in side["tok_keys"]}
        side["block_buffer"] = {k: [] for k in side["tok_keys"]}


def compact_side(side):
    if side["offset"] <= 0 or not side["carry"]:
        return
    off = side["offset"]
    for key in side["tok_keys"]:
        side["carry"][key] = side["carry"][key][off:]
    side["offset"] = 0


def flush_side_chunk(side_name, side, chunks_root):
    if not side["tok_keys"]:
        return 0
    any_key = side["tok_keys"][0]
    n_blocks = len(side["block_buffer"][any_key])
    if n_blocks == 0:
        return 0
    chunk_dir = os.path.join(chunks_root, f"{side_name}_chunk_{side['chunks_written']:06d}")
    ds = Dataset.from_dict(side["block_buffer"])
    ds.save_to_disk(chunk_dir)
    for key in side["tok_keys"]:
        side["block_buffer"][key].clear()
    side["chunks_written"] += 1
    return n_blocks


def add_tokens_to_side(side, tok, block_size, token_target):
    if side["written_tokens"] >= token_target:
        return 0
    ensure_side_keys(side, tok)

    for key in side["tok_keys"]:
        values = tok.get(key, [])
        if values and isinstance(values[0], list):
            values = values[0]
        side["carry"][key].extend(values)

    any_key = side["tok_keys"][0]
    new_blocks = 0
    while (len(side["carry"][any_key]) - side["offset"]) >= block_size and side["written_tokens"] < token_target:
        if side["written_tokens"] + block_size > token_target:
            break
        start = side["offset"]
        end = start + block_size
        for key in side["tok_keys"]:
            side["block_buffer"][key].append(side["carry"][key][start:end])
        side["offset"] += block_size
        side["written_blocks"] += 1
        side["written_tokens"] += block_size
        new_blocks += 1

    return new_blocks


def merge_chunk_side(chunks_root, side_name):
    chunk_paths = sorted(
        [
            os.path.join(chunks_root, d)
            for d in os.listdir(chunks_root)
            if d.startswith(f"{side_name}_chunk_") and os.path.isdir(os.path.join(chunks_root, d))
        ]
    )
    if not chunk_paths:
        return None, []
    datasets = [load_from_disk(p) for p in chunk_paths]
    merged = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    return merged, chunk_paths


def maybe_add_sample(samples, category, record, sample_size):
    bucket = samples[category]
    if len(bucket) < sample_size:
        bucket.append(record)


def sample_payload(idx, text, signals, strict_types):
    return {
        "index": int(idx),
        "filters_triggered": list(signals["filters_triggered"]),
        "filter_ql_type": signals["ql_type"],
        "keyword_counts": dict(signals["keyword_counts"]),
        "strict_query_types": list(strict_types),
        "text_preview": compact_text(text),
    }


def main():
    args = parse_args()

    if args.dirty_rate < 0.0 or args.dirty_rate > 1.0:
        raise ValueError(f"--dirty_rate must be in [0, 1], got {args.dirty_rate}")
    if args.dirty_pool_multiplier < 1.0:
        raise ValueError(f"--dirty_pool_multiplier must be >= 1.0, got {args.dirty_pool_multiplier}")
    if args.clean_pool_multiplier < 1.0:
        raise ValueError(f"--clean_pool_multiplier must be >= 1.0, got {args.clean_pool_multiplier}")

    target_tokens = resolve_target_tokens(args)
    model_name = resolve_model_name(args)

    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")
    if target_tokens % args.block_size != 0:
        raise ValueError(
            f"target_tokens={target_tokens} must be divisible by block_size={args.block_size}"
        )

    target_blocks = target_tokens // args.block_size
    dirty_target_blocks = int(round(target_blocks * args.dirty_rate))
    if args.dirty_rate > 0.0 and dirty_target_blocks == 0:
        dirty_target_blocks = 1
    if args.dirty_rate < 1.0 and dirty_target_blocks == target_blocks:
        dirty_target_blocks = target_blocks - 1
    clean_target_blocks = target_blocks - dirty_target_blocks

    dirty_pool_target_blocks = int(round(dirty_target_blocks * args.dirty_pool_multiplier))
    dirty_pool_target_blocks = max(dirty_pool_target_blocks, dirty_target_blocks)
    clean_pool_target_blocks = int(round(clean_target_blocks * args.clean_pool_multiplier))
    clean_pool_target_blocks = max(clean_pool_target_blocks, clean_target_blocks)
    clean_target_tokens = clean_target_blocks * args.block_size
    clean_pool_target_tokens = clean_pool_target_blocks * args.block_size
    dirty_target_tokens = dirty_target_blocks * args.block_size
    dirty_pool_target_tokens = dirty_pool_target_blocks * args.block_size

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    chunks_root = os.path.join(args.output_dir, "grouped_chunks")
    os.makedirs(chunks_root, exist_ok=True)
    progress_path = os.path.join(args.output_dir, "progress.json")

    print(f"Target total tokens: {target_tokens:,} ({target_blocks:,} blocks)")
    print(f"Target clean tokens: {clean_target_tokens:,} ({clean_target_blocks:,} blocks)")
    print(
        f"Clean pool tokens: {clean_pool_target_tokens:,} ({clean_pool_target_blocks:,} blocks) "
        f"| multiplier={args.clean_pool_multiplier}"
    )
    print(f"Target dirty tokens: {dirty_target_tokens:,} ({dirty_target_blocks:,} blocks)")
    print(
        f"Dirty pool tokens: {dirty_pool_target_tokens:,} ({dirty_pool_target_blocks:,} blocks) "
        f"| mode={args.dirty_selection_mode}"
    )
    print(f"Tokenizer: {model_name}")

    tokenizer = load_tokenizer(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    clean_side = init_side()
    dirty_side = init_side()
    source_state = {
        "source_rows_seen": 0,
        "source_rows_kept_clean_candidate": 0,
        "source_rows_kept_dirty_candidate": 0,
        "source_rows_filtered_out": 0,
        "filtered_keyword_only": 0,
        "filtered_keyword_plus_regex": 0,
        "filtered_regex_only": 0,
    }

    # Audit stats modeled after 04_analyze_filtering.py.
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
    samples = {
        "dirty_regex": [],
        "dirty_keyword_only": [],
        "dirty_keyword_only_ambiguous": [],
        "dirty_keyword_only_no_strict": [],
        "clean_strict_leakage": [],
        "dirty_selected_pool": [],
    }

    if args.resume:
        progress = load_progress(progress_path)
        if progress is not None:
            source_state.update(progress.get("source_state", {}))
            clean_side.update(progress.get("clean_side", {}))
            dirty_side.update(progress.get("dirty_side", {}))
            totals.update(progress.get("audit", {}).get("totals", {}))
            trigger_breakdown.update(progress.get("audit", {}).get("trigger_breakdown", {}))
            filter_ql_types.update(progress.get("audit", {}).get("filter_ql_types", {}))
            strict_in_dirty.update(progress.get("audit", {}).get("strict_in_dirty", {}))
            strict_in_clean.update(progress.get("audit", {}).get("strict_in_clean", {}))
            keyword_docfreq_dirty.update(progress.get("audit", {}).get("keyword_docfreq_dirty", {}))
            keyword_occ_dirty.update(progress.get("audit", {}).get("keyword_occ_dirty", {}))
            keyword_docfreq_keyword_only.update(progress.get("audit", {}).get("keyword_docfreq_keyword_only", {}))
            keyword_occ_keyword_only.update(progress.get("audit", {}).get("keyword_occ_keyword_only", {}))
            keyword_combo_dirty.update(progress.get("audit", {}).get("keyword_combo_dirty", {}))
            keyword_combo_keyword_only.update(progress.get("audit", {}).get("keyword_combo_keyword_only", {}))
            csc = progress.get("audit", {}).get("char_stats_clean", {})
            csd = progress.get("audit", {}).get("char_stats_dirty", {})
            if csc:
                char_stats_clean.n = csc.get("n", 0)
                char_stats_clean.total = csc.get("total", 0)
                char_stats_clean.min_val = csc.get("min", None)
                char_stats_clean.max_val = csc.get("max", None)
            if csd:
                char_stats_dirty.n = csd.get("n", 0)
                char_stats_dirty.total = csd.get("total", 0)
                char_stats_dirty.min_val = csd.get("min", None)
                char_stats_dirty.max_val = csd.get("max", None)
            samples = progress.get("samples", samples)
            print(
                f"Resuming: seen={source_state['source_rows_seen']:,} "
                f"clean_tokens={clean_side['written_tokens']:,} dirty_pool_tokens={dirty_side['written_tokens']:,}"
            )

    def save_progress():
        write_json(
            progress_path,
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "source_state": source_state,
                "clean_side": clean_side,
                "dirty_side": dirty_side,
                "audit": {
                    "totals": dict(totals),
                    "trigger_breakdown": dict(trigger_breakdown),
                    "filter_ql_types": dict(filter_ql_types),
                    "strict_in_dirty": dict(strict_in_dirty),
                    "strict_in_clean": dict(strict_in_clean),
                    "keyword_docfreq_dirty": dict(keyword_docfreq_dirty),
                    "keyword_occ_dirty": dict(keyword_occ_dirty),
                    "keyword_docfreq_keyword_only": dict(keyword_docfreq_keyword_only),
                    "keyword_occ_keyword_only": dict(keyword_occ_keyword_only),
                    "keyword_combo_dirty": dict(keyword_combo_dirty),
                    "keyword_combo_keyword_only": dict(keyword_combo_keyword_only),
                    "char_stats_clean": char_stats_clean.to_dict(),
                    "char_stats_dirty": char_stats_dirty.to_dict(),
                },
                "samples": samples,
                "targets": {
                    "target_tokens": target_tokens,
                    "clean_target_tokens": clean_target_tokens,
                    "clean_pool_target_tokens": clean_pool_target_tokens,
                    "dirty_target_tokens": dirty_target_tokens,
                    "dirty_pool_target_tokens": dirty_pool_target_tokens,
                    "block_size": args.block_size,
                },
            },
        )

    print(f"Loading streaming dataset: {args.source_dataset}/{args.source_config} split={args.source_split}")
    stream = load_dataset(
        args.source_dataset,
        args.source_config,
        split=args.source_split,
        streaming=True,
        cache_dir=args.cache_dir,
    )
    if source_state["source_rows_seen"] > 0:
        print(f"Skipping already-seen rows: {source_state['source_rows_seen']:,}")
        stream = stream.skip(source_state["source_rows_seen"])

    clean_pbar = tqdm(
        total=clean_pool_target_tokens,
        initial=clean_side["written_tokens"],
        desc="Collect clean pool tokens",
        ncols=100,
    )
    dirty_pbar = tqdm(
        total=dirty_pool_target_tokens,
        initial=dirty_side["written_tokens"],
        desc="Collect dirty pool tokens",
        ncols=100,
    )

    for sample in stream:
        if args.max_source_rows is not None and source_state["source_rows_seen"] >= args.max_source_rows:
            print(f"Reached max_source_rows={args.max_source_rows}, stopping.")
            break
        if clean_side["written_tokens"] >= clean_pool_target_tokens and dirty_side["written_tokens"] >= dirty_pool_target_tokens:
            break

        idx = source_state["source_rows_seen"]
        source_state["source_rows_seen"] += 1
        text = sample["text"]

        signals = detect_filter_signals(text)
        strict_types = detect_strict_query_types(text)

        has_keyword = "keyword" in signals["filters_triggered"]
        has_regex = "regex" in signals["filters_triggered"]
        is_filtered = has_keyword or has_regex
        has_strict = bool(strict_types)
        keywords_present = sorted(signals["keyword_counts"].keys())
        key_combo = "+".join(keywords_present) if keywords_present else "(none)"

        # Audit counters.
        totals["total"] += 1
        if is_filtered:
            totals["dirty"] += 1
            char_stats_dirty.add(len(text))
            if has_keyword and has_regex:
                trigger_breakdown["keyword+regex"] += 1
                totals["dirty_keyword_plus_regex"] += 1
            elif has_keyword:
                trigger_breakdown["keyword_only"] += 1
                totals["dirty_keyword_only"] += 1
            else:
                trigger_breakdown["regex_only"] += 1
                totals["dirty_regex_only"] += 1
            if has_regex:
                filter_ql_types[signals["ql_type"] or "unknown"] += 1
                maybe_add_sample(samples, "dirty_regex", sample_payload(idx, text, signals, strict_types), args.sample_size)
            if has_keyword and not has_regex:
                maybe_add_sample(samples, "dirty_keyword_only", sample_payload(idx, text, signals, strict_types), args.sample_size)
                if set(keywords_present).issubset(AMBIGUOUS_QUERY_KEYWORDS):
                    totals["dirty_keyword_only_ambiguous"] += 1
                    maybe_add_sample(
                        samples,
                        "dirty_keyword_only_ambiguous",
                        sample_payload(idx, text, signals, strict_types),
                        args.sample_size,
                    )
            if has_keyword:
                keyword_combo_dirty[key_combo] += 1
                for kw, occ in signals["keyword_counts"].items():
                    keyword_docfreq_dirty[kw] += 1
                    keyword_occ_dirty[kw] += int(occ)
            if has_keyword and not has_regex:
                keyword_combo_keyword_only[key_combo] += 1
                for kw, occ in signals["keyword_counts"].items():
                    keyword_docfreq_keyword_only[kw] += 1
                    keyword_occ_keyword_only[kw] += int(occ)
            if has_strict:
                totals["dirty_strict_query_docs"] += 1
                for lang in strict_types:
                    strict_in_dirty[lang] += 1
            else:
                totals["dirty_no_strict_query_docs"] += 1
                if has_keyword and not has_regex:
                    maybe_add_sample(
                        samples,
                        "dirty_keyword_only_no_strict",
                        sample_payload(idx, text, signals, strict_types),
                        args.sample_size,
                    )
        else:
            totals["clean"] += 1
            trigger_breakdown["none"] += 1
            char_stats_clean.add(len(text))
            if has_strict:
                totals["clean_strict_query_docs"] += 1
                for lang in strict_types:
                    strict_in_clean[lang] += 1
                maybe_add_sample(
                    samples,
                    "clean_strict_leakage",
                    sample_payload(idx, text, signals, strict_types),
                    args.sample_size,
                )

        # Selection logic.
        is_clean_candidate = not is_filtered
        if args.clean_exclude_strict and has_strict:
            is_clean_candidate = False

        if args.dirty_selection_mode == "regex":
            is_dirty_candidate = has_regex
        elif args.dirty_selection_mode == "strict":
            is_dirty_candidate = has_strict
        elif args.dirty_selection_mode == "regex_or_strict":
            is_dirty_candidate = has_regex or has_strict
        else:
            is_dirty_candidate = is_filtered

        need_clean = is_clean_candidate and clean_side["written_tokens"] < clean_pool_target_tokens
        need_dirty = is_dirty_candidate and dirty_side["written_tokens"] < dirty_pool_target_tokens
        if not (need_clean or need_dirty):
            if is_filtered:
                source_state["source_rows_filtered_out"] += 1
                if has_keyword and has_regex:
                    source_state["filtered_keyword_plus_regex"] += 1
                elif has_keyword:
                    source_state["filtered_keyword_only"] += 1
                elif has_regex:
                    source_state["filtered_regex_only"] += 1
            if source_state["source_rows_seen"] % args.log_every_rows == 0:
                save_progress()
            continue

        if need_clean:
            source_state["source_rows_kept_clean_candidate"] += 1
            clean_side["source_rows_used"] += 1
        if need_dirty:
            source_state["source_rows_kept_dirty_candidate"] += 1
            dirty_side["source_rows_used"] += 1
            maybe_add_sample(
                samples,
                "dirty_selected_pool",
                sample_payload(idx, text, signals, strict_types),
                args.sample_size,
            )

        if is_filtered:
            source_state["source_rows_filtered_out"] += 1
            if has_keyword and has_regex:
                source_state["filtered_keyword_plus_regex"] += 1
            elif has_keyword:
                source_state["filtered_keyword_only"] += 1
            elif has_regex:
                source_state["filtered_regex_only"] += 1

        tok = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=True,
        )

        if need_clean:
            old_tokens = clean_side["written_tokens"]
            add_tokens_to_side(clean_side, tok, args.block_size, clean_pool_target_tokens)
            clean_pbar.update(clean_side["written_tokens"] - old_tokens)
        if need_dirty:
            old_tokens = dirty_side["written_tokens"]
            add_tokens_to_side(dirty_side, tok, args.block_size, dirty_pool_target_tokens)
            dirty_pbar.update(dirty_side["written_tokens"] - old_tokens)

        for side_name, side in (("clean", clean_side), ("dirty", dirty_side)):
            if side["tok_keys"]:
                any_key = side["tok_keys"][0]
                if len(side["block_buffer"][any_key]) >= args.chunk_blocks:
                    compact_side(side)
                    flush_side_chunk(side_name, side, chunks_root)
                    save_progress()
                if side["offset"] >= 200000:
                    compact_side(side)

        if source_state["source_rows_seen"] % args.log_every_rows == 0:
            seen = source_state["source_rows_seen"]
            filt = source_state["source_rows_filtered_out"]
            print(
                f"seen={seen:,} filtered={filt:,} ({pct(filt, seen):.2f}%) "
                f"clean_pool_tokens={clean_side['written_tokens']:,}/{clean_pool_target_tokens:,} "
                f"dirty_pool_tokens={dirty_side['written_tokens']:,}/{dirty_pool_target_tokens:,}"
            )
            save_progress()

    clean_pbar.close()
    dirty_pbar.close()

    # Final flush.
    for side_name, side in (("clean", clean_side), ("dirty", dirty_side)):
        if side["tok_keys"]:
            compact_side(side)
            flush_side_chunk(side_name, side, chunks_root)

    done = clean_side["written_tokens"] >= clean_pool_target_tokens and dirty_side["written_tokens"] >= dirty_pool_target_tokens
    print("Done target reached." if done else "Stopped before reaching full targets.")

    # Merge clean and dirty pools.
    merged_clean, clean_chunk_paths = merge_chunk_side(chunks_root, "clean")
    merged_dirty_pool, dirty_chunk_paths = merge_chunk_side(chunks_root, "dirty")

    if merged_clean is None:
        raise RuntimeError("No clean chunks were written; cannot create final_clean_grouped.")
    if merged_dirty_pool is None and dirty_target_tokens > 0:
        raise RuntimeError("No dirty chunks were written; cannot create final_dirtymatched_grouped.")

    final_clean_path = os.path.join(args.output_dir, "final_clean_grouped")
    final_clean_pool_path = os.path.join(args.output_dir, "final_clean_pool_grouped")
    final_dirty_pool_path = os.path.join(args.output_dir, "final_dirty_pool_grouped")
    final_dirtymatched_path = os.path.join(args.output_dir, "final_dirtymatched_grouped")

    if os.path.exists(final_clean_path):
        shutil.rmtree(final_clean_path)
    if os.path.exists(final_clean_pool_path):
        shutil.rmtree(final_clean_pool_path)
    if os.path.exists(final_dirty_pool_path):
        shutil.rmtree(final_dirty_pool_path)
    if os.path.exists(final_dirtymatched_path):
        shutil.rmtree(final_dirtymatched_path)

    if clean_pool_target_blocks > clean_target_blocks:
        merged_clean.save_to_disk(final_clean_pool_path)
        final_clean = merged_clean.select(range(clean_target_blocks))
        final_clean.save_to_disk(final_clean_path)
        clean_pool_path_for_summary = final_clean_pool_path
    else:
        final_clean = merged_clean
        if len(final_clean) > clean_target_blocks:
            final_clean = final_clean.select(range(clean_target_blocks))
        final_clean.save_to_disk(final_clean_path)
        clean_pool_path_for_summary = final_clean_path

    if dirty_target_tokens > 0:
        merged_dirty_pool.save_to_disk(final_dirty_pool_path)
        final_dirtymatched = merged_dirty_pool
        if len(final_dirtymatched) > dirty_target_blocks:
            final_dirtymatched = final_dirtymatched.select(range(dirty_target_blocks))
        final_dirtymatched.save_to_disk(final_dirtymatched_path)
    else:
        final_dirtymatched = None
        final_dirty_pool_path = None

    # Optional shared dev linkage/copy.
    if args.shared_dev_mode != "skip" and args.shared_dev_from:
        if os.path.exists(args.shared_dev_from):
            shared_dev_group_dir = os.path.join(args.output_dir, "shared_dev_grouped")
            os.makedirs(shared_dev_group_dir, exist_ok=True)
            shared_dev_target = os.path.join(shared_dev_group_dir, "grouped")
            if os.path.lexists(shared_dev_target):
                if os.path.islink(shared_dev_target) or os.path.isfile(shared_dev_target):
                    os.unlink(shared_dev_target)
                else:
                    shutil.rmtree(shared_dev_target)
            if args.shared_dev_mode == "copy":
                shutil.copytree(args.shared_dev_from, shared_dev_target)
            else:
                os.symlink(args.shared_dev_from, shared_dev_target)
        else:
            print(f"Warning: shared_dev_from does not exist, skipping: {args.shared_dev_from}")

    # Build audit summary sections.
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

    filtering_audit = {
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
            "clean": char_stats_clean.to_dict(),
            "dirty": char_stats_dirty.to_dict(),
        },
        "top_keyword_combos_dirty": combo_rows_dirty[:25],
        "top_keyword_combos_keyword_only": combo_rows_keyword_only[:25],
    }

    summary = {
        "run_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": args.source_dataset,
            "source_config": args.source_config,
            "source_split": args.source_split,
            "cache_dir": args.cache_dir,
            "target_ds": args.target_ds,
            "target_tokens": int(target_tokens),
            "block_size": args.block_size,
            "dirty_rate_target": args.dirty_rate,
            "dirty_selection_mode": args.dirty_selection_mode,
            "clean_exclude_strict": args.clean_exclude_strict,
            "dirty_pool_multiplier": args.dirty_pool_multiplier,
            "clean_pool_multiplier": args.clean_pool_multiplier,
            "output_dir": args.output_dir,
            "model_name_or_path": model_name,
        },
        "targets": {
            "target_blocks": int(target_blocks),
            "clean_target_blocks": int(clean_target_blocks),
            "clean_pool_target_blocks": int(clean_pool_target_blocks),
            "dirty_target_blocks": int(dirty_target_blocks),
            "dirty_pool_target_blocks": int(dirty_pool_target_blocks),
            "clean_target_tokens": int(clean_target_tokens),
            "clean_pool_target_tokens": int(clean_pool_target_tokens),
            "dirty_target_tokens": int(dirty_target_tokens),
            "dirty_pool_target_tokens": int(dirty_pool_target_tokens),
        },
        "collection_state": {
            **source_state,
            "clean_written_blocks": int(clean_side["written_blocks"]),
            "clean_written_tokens": int(clean_side["written_tokens"]),
            "clean_chunks_written": int(clean_side["chunks_written"]),
            "dirty_pool_written_blocks": int(dirty_side["written_blocks"]),
            "dirty_pool_written_tokens": int(dirty_side["written_tokens"]),
            "dirty_chunks_written": int(dirty_side["chunks_written"]),
        },
        "final_datasets": {
            "final_clean_grouped_path": final_clean_path,
            "final_clean_num_blocks": int(len(final_clean)),
            "final_clean_num_tokens": int(len(final_clean) * args.block_size),
            "final_clean_pool_grouped_path": clean_pool_path_for_summary,
            "final_clean_pool_num_blocks": int(len(merged_clean)),
            "final_clean_pool_num_tokens": int(len(merged_clean) * args.block_size),
            "final_dirty_pool_grouped_path": final_dirty_pool_path,
            "final_dirty_pool_num_blocks": int(len(merged_dirty_pool)) if merged_dirty_pool is not None else 0,
            "final_dirty_pool_num_tokens": int(len(merged_dirty_pool) * args.block_size) if merged_dirty_pool is not None else 0,
            "final_dirtymatched_grouped_path": final_dirtymatched_path if final_dirtymatched is not None else None,
            "final_dirtymatched_num_blocks": int(len(final_dirtymatched)) if final_dirtymatched is not None else 0,
            "final_dirtymatched_num_tokens": int(len(final_dirtymatched) * args.block_size) if final_dirtymatched is not None else 0,
            "final_dirtymatched_rate_pct_of_total": pct(
                int(len(final_dirtymatched) * args.block_size) if final_dirtymatched is not None else 0,
                int(len(final_clean) * args.block_size) + (
                    int(len(final_dirtymatched) * args.block_size) if final_dirtymatched is not None else 0
                ),
            ),
            "clean_chunk_count": len(clean_chunk_paths),
            "dirty_chunk_count": len(dirty_chunk_paths),
        },
        "filtering_audit": filtering_audit,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    write_json(summary_path, summary)
    write_json(
        os.path.join(args.output_dir, "final_stats.json"),
        {
            "final_clean": {
                "num_blocks": int(len(final_clean)),
                "num_tokens": int(len(final_clean) * args.block_size),
            },
            "final_clean_pool": {
                "num_blocks": int(len(merged_clean)),
                "num_tokens": int(len(merged_clean) * args.block_size),
            },
            "final_dirtymatched": {
                "num_blocks": int(len(final_dirtymatched)) if final_dirtymatched is not None else 0,
                "num_tokens": int(len(final_dirtymatched) * args.block_size) if final_dirtymatched is not None else 0,
            },
            "match_ratio_dirty_to_clean": (
                round((len(final_dirtymatched) / len(final_clean)), 6)
                if (final_dirtymatched is not None and len(final_clean) > 0)
                else 0.0
            ),
        },
    )

    write_csv(
        os.path.join(args.output_dir, "keyword_trigger_stats.csv"),
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

    for category, records in samples.items():
        write_jsonl(os.path.join(args.output_dir, f"samples_{category}.jsonl"), records)

    manual_rows = []
    for category, records in samples.items():
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
        os.path.join(args.output_dir, "manual_review_samples.csv"),
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

    save_progress()
    print(f"Saved summary: {summary_path}")
    print(f"Saved final_clean_grouped: {final_clean_path}")
    if clean_pool_target_blocks > clean_target_blocks:
        print(f"Saved final_clean_pool_grouped: {final_clean_pool_path}")
    if final_dirtymatched is not None:
        print(f"Saved final_dirtymatched_grouped: {final_dirtymatched_path}")
        print(f"Saved final_dirty_pool_grouped: {final_dirty_pool_path}")


if __name__ == "__main__":
    main()
