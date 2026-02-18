import argparse
import json
import os
import shutil
from datetime import datetime, timezone

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import detect_filter_signals, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream C4 (or another dataset), aggressively filter query-like docs, "
            "and keep grouped token blocks until target token count is reached."
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
        help="Config used to infer target tokens and tokenizer when not explicitly provided.",
    )
    parser.add_argument("--target_tokens", type=int, default=None)
    parser.add_argument("--target_stats_path", type=str, default=None)
    parser.add_argument("--target_stats_field", type=str, default="final_clean.num_tokens")

    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--chunk_blocks", type=int, default=20000)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work/datasets/c4-en-noclean-match-owt-clean",
        help="Will contain grouped chunk datasets and progress/summary JSON files.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from output_dir/progress.json if present.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, delete output_dir before starting.",
    )
    parser.add_argument("--max_source_rows", type=int, default=None)
    parser.add_argument("--log_every_rows", type=int, default=10000)
    return parser.parse_args()


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
    value = int(get_nested_value(stats, args.target_stats_field))
    return value


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
                "or pre-download the tokenizer to cache."
            ) from e


def load_progress(progress_path):
    if not os.path.exists(progress_path):
        return None
    with open(progress_path) as f:
        return json.load(f)


def compact_carry(carry, offset):
    if not carry:
        return carry, 0
    if offset <= 0:
        return carry, 0
    for key in list(carry.keys()):
        carry[key] = carry[key][offset:]
    return carry, 0


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def flush_chunk(output_chunks_dir, chunk_idx, block_buffer):
    if not block_buffer:
        return 0
    any_key = next(iter(block_buffer.keys()))
    n_blocks = len(block_buffer[any_key])
    if n_blocks == 0:
        return 0
    chunk_dir = os.path.join(output_chunks_dir, f"chunk_{chunk_idx:06d}")
    ds = Dataset.from_dict(block_buffer)
    ds.save_to_disk(chunk_dir)
    for key in block_buffer.keys():
        block_buffer[key].clear()
    return n_blocks


def main():
    args = parse_args()

    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    output_chunks_dir = os.path.join(args.output_dir, "grouped_chunks")
    os.makedirs(output_chunks_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, "progress.json")
    summary_path = os.path.join(args.output_dir, "summary.json")

    target_tokens = resolve_target_tokens(args)
    model_name = resolve_model_name(args)

    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0, got {target_tokens}")
    if target_tokens % args.block_size != 0:
        raise ValueError(
            f"target_tokens={target_tokens} is not divisible by block_size={args.block_size}. "
            "Provide a compatible value."
        )

    print(f"Target tokens: {target_tokens}")
    print(f"Target blocks: {target_tokens // args.block_size}")
    print(f"Tokenizer: {model_name}")

    tokenizer = load_tokenizer(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    state = {
        "source_rows_seen": 0,
        "source_rows_kept_clean": 0,
        "source_rows_filtered_out": 0,
        "filtered_keyword_only": 0,
        "filtered_keyword_plus_regex": 0,
        "filtered_regex_only": 0,
        "written_blocks": 0,
        "written_tokens": 0,
        "chunks_written": 0,
    }
    carry = {}
    offset = 0
    tok_keys = None
    block_buffer = {}

    if args.resume:
        progress = load_progress(progress_path)
        if progress is not None:
            state.update(progress.get("state", {}))
            carry = progress.get("carry", {})
            tok_keys = progress.get("tok_keys")
            offset = int(progress.get("offset", 0))
            print(
                f"Resuming from progress: seen={state['source_rows_seen']}, "
                f"written_tokens={state['written_tokens']}, chunks={state['chunks_written']}"
            )
            if tok_keys is not None and not block_buffer:
                block_buffer = {k: [] for k in tok_keys}

    print(
        f"Loading streaming dataset: {args.source_dataset} / {args.source_config} / {args.source_split}"
    )
    stream = load_dataset(
        args.source_dataset,
        args.source_config,
        split=args.source_split,
        streaming=True,
        cache_dir=args.cache_dir,
    )

    if state["source_rows_seen"] > 0:
        print(f"Skipping already-seen rows: {state['source_rows_seen']}")
        stream = stream.skip(state["source_rows_seen"])

    pbar = tqdm(total=target_tokens, initial=state["written_tokens"], desc="Writing target tokens", ncols=100)

    for sample in stream:
        if args.max_source_rows is not None and state["source_rows_seen"] >= args.max_source_rows:
            print(f"Reached max_source_rows={args.max_source_rows}, stopping early.")
            break
        if state["written_tokens"] >= target_tokens:
            break

        text = sample.get("text", "")
        state["source_rows_seen"] += 1

        signals = detect_filter_signals(text)
        has_keyword = "keyword" in signals["filters_triggered"]
        has_regex = "regex" in signals["filters_triggered"]
        is_filtered = has_keyword or has_regex

        if is_filtered:
            state["source_rows_filtered_out"] += 1
            if has_keyword and has_regex:
                state["filtered_keyword_plus_regex"] += 1
            elif has_keyword:
                state["filtered_keyword_only"] += 1
            elif has_regex:
                state["filtered_regex_only"] += 1
        else:
            state["source_rows_kept_clean"] += 1
            tok = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=True,
            )

            if tok_keys is None:
                tok_keys = sorted(tok.keys())
                carry = {k: [] for k in tok_keys}
                block_buffer = {k: [] for k in tok_keys}

            for key in tok_keys:
                values = tok.get(key, [])
                if values and isinstance(values[0], list):
                    values = values[0]
                carry[key].extend(values)

            first_key = tok_keys[0]
            while (len(carry[first_key]) - offset) >= args.block_size and state["written_tokens"] < target_tokens:
                if state["written_tokens"] + args.block_size > target_tokens:
                    break
                for key in tok_keys:
                    block_buffer[key].append(carry[key][offset : offset + args.block_size])
                offset += args.block_size
                state["written_blocks"] += 1
                state["written_tokens"] += args.block_size
                pbar.update(args.block_size)

                if len(block_buffer[first_key]) >= args.chunk_blocks:
                    carry, offset = compact_carry(carry, offset)
                    n_saved = flush_chunk(output_chunks_dir, state["chunks_written"], block_buffer)
                    state["chunks_written"] += 1 if n_saved > 0 else 0
                    write_json(
                        progress_path,
                        {
                            "state": state,
                            "tok_keys": tok_keys,
                            "carry": carry,
                            "offset": offset,
                            "updated_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )

            if offset >= 200000:
                carry, offset = compact_carry(carry, offset)

        if state["source_rows_seen"] % args.log_every_rows == 0:
            dirty = state["source_rows_filtered_out"]
            seen = state["source_rows_seen"]
            dirty_pct = (100.0 * dirty / seen) if seen else 0.0
            print(
                f"seen={seen:,} kept={state['source_rows_kept_clean']:,} "
                f"filtered={dirty:,} ({dirty_pct:.2f}%) written_tokens={state['written_tokens']:,}"
            )

    pbar.close()

    if tok_keys is not None and any(len(v) > 0 for v in block_buffer.values()):
        carry, offset = compact_carry(carry, offset)
        n_saved = flush_chunk(output_chunks_dir, state["chunks_written"], block_buffer)
        state["chunks_written"] += 1 if n_saved > 0 else 0

    done = state["written_tokens"] >= target_tokens
    if done:
        print("Target token budget reached.")
    else:
        print("Stopped before hitting target token budget.")

    write_json(
        progress_path,
        {
            "state": state,
            "tok_keys": tok_keys,
            "carry": carry,
            "offset": offset,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "done": done,
        },
    )

    summary = {
        "run_meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": args.source_dataset,
            "source_config": args.source_config,
            "source_split": args.source_split,
            "target_ds": args.target_ds,
            "target_tokens": target_tokens,
            "block_size": args.block_size,
            "chunk_blocks": args.chunk_blocks,
            "output_dir": args.output_dir,
            "model_name_or_path": model_name,
            "resume": args.resume,
            "max_source_rows": args.max_source_rows,
        },
        "state": state,
        "completion": {
            "done": done,
            "remaining_tokens": max(0, target_tokens - state["written_tokens"]),
        },
        "filter_stats": {
            "filtered_keyword_only": state["filtered_keyword_only"],
            "filtered_keyword_plus_regex": state["filtered_keyword_plus_regex"],
            "filtered_regex_only": state["filtered_regex_only"],
            "filtered_total": state["source_rows_filtered_out"],
            "filtered_rate_pct_over_seen": round(
                (100.0 * state["source_rows_filtered_out"] / state["source_rows_seen"])
                if state["source_rows_seen"]
                else 0.0,
                4,
            ),
        },
    }
    write_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")
    print(f"Saved progress: {progress_path}")
    print(f"Grouped chunks dir: {output_chunks_dir}")


if __name__ == "__main__":
    main()
