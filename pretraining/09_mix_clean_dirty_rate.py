import argparse
import json
import os
import shutil
from datetime import datetime, timezone

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a new clean/dirty mixture at a chosen dirty rate from already-saved "
            "grouped clean and dirty pool datasets (no re-streaming)."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Directory containing final_clean_grouped and final_dirty_pool_grouped "
            "(produced by 08_stream_build_clean_dirty_with_stats.py)."
        ),
    )
    parser.add_argument("--dirty_rate", type=float, required=True, help="Target dirty fraction in [0, 1].")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=None,
        help=(
            "Total token budget of the mixed dataset. Defaults to "
            "(current final_clean + current final_dirtymatched) from input_dir/final_stats.json."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for new final_clean_grouped/final_dirtymatched_grouped. Defaults to input_dir.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix for output dirs, e.g. dirty10p -> final_clean_grouped_dirty10p.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()

    if args.dirty_rate < 0.0 or args.dirty_rate > 1.0:
        raise ValueError(f"--dirty_rate must be in [0, 1], got {args.dirty_rate}")

    clean_pool_path = os.path.join(args.input_dir, "final_clean_pool_grouped")
    if not os.path.exists(clean_pool_path):
        clean_pool_path = os.path.join(args.input_dir, "final_clean_grouped")
    dirty_pool_path = os.path.join(args.input_dir, "final_dirty_pool_grouped")
    if not os.path.exists(clean_pool_path):
        raise FileNotFoundError(f"Missing clean pool dataset: {clean_pool_path}")
    if not os.path.exists(dirty_pool_path):
        raise FileNotFoundError(f"Missing dirty pool dataset: {dirty_pool_path}")

    clean_pool = load_from_disk(clean_pool_path)
    dirty_pool = load_from_disk(dirty_pool_path)

    if args.total_tokens is not None:
        total_tokens = int(args.total_tokens)
    else:
        stats_path = os.path.join(args.input_dir, "final_stats.json")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Missing --total_tokens and missing {stats_path}; cannot infer total token budget."
            )
        with open(stats_path) as f:
            stats = json.load(f)
        total_tokens = int(stats["final_clean"]["num_tokens"] + stats["final_dirtymatched"]["num_tokens"])

    if total_tokens % args.block_size != 0:
        raise ValueError(
            f"total_tokens={total_tokens} must be divisible by block_size={args.block_size}"
        )

    total_blocks = total_tokens // args.block_size
    dirty_blocks = int(round(total_blocks * args.dirty_rate))
    if args.dirty_rate > 0.0 and dirty_blocks == 0:
        dirty_blocks = 1
    if args.dirty_rate < 1.0 and dirty_blocks == total_blocks:
        dirty_blocks = total_blocks - 1
    clean_blocks = total_blocks - dirty_blocks

    if clean_blocks > len(clean_pool):
        raise ValueError(
            f"Requested clean_blocks={clean_blocks}, but clean pool has only {len(clean_pool)} blocks. "
            "Rebuild pool with more clean tokens."
        )
    if dirty_blocks > len(dirty_pool):
        raise ValueError(
            f"Requested dirty_blocks={dirty_blocks}, but dirty pool has only {len(dirty_pool)} blocks. "
            "Rebuild pool with larger dirty_pool_multiplier or smaller total_tokens/dirty_rate."
        )

    mixed_clean = clean_pool.select(range(clean_blocks))
    mixed_dirty = dirty_pool.select(range(dirty_blocks))

    out_root = args.output_dir or args.input_dir
    os.makedirs(out_root, exist_ok=True)

    suffix = f"_{args.suffix}" if args.suffix else ""
    out_clean_path = os.path.join(out_root, f"final_clean_grouped{suffix}")
    out_dirty_path = os.path.join(out_root, f"final_dirtymatched_grouped{suffix}")

    for p in (out_clean_path, out_dirty_path):
        if os.path.exists(p):
            if args.overwrite:
                shutil.rmtree(p)
            else:
                raise FileExistsError(f"{p} already exists. Use --overwrite to replace.")

    mixed_clean.save_to_disk(out_clean_path)
    mixed_dirty.save_to_disk(out_dirty_path)

    mix_stats = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": args.input_dir,
        "output_dir": out_root,
        "dirty_rate_target": args.dirty_rate,
        "total_tokens": int(total_tokens),
        "block_size": args.block_size,
        "total_blocks": int(total_blocks),
        "final_clean": {
            "path": out_clean_path,
            "num_blocks": int(len(mixed_clean)),
            "num_tokens": int(len(mixed_clean) * args.block_size),
        },
        "final_dirtymatched": {
            "path": out_dirty_path,
            "num_blocks": int(len(mixed_dirty)),
            "num_tokens": int(len(mixed_dirty) * args.block_size),
        },
        "final_dirtymatched_rate_pct_of_total": round(
            100.0 * len(mixed_dirty) / total_blocks if total_blocks else 0.0, 4
        ),
    }

    stats_suffix = f"_{args.suffix}" if args.suffix else ""
    stats_path = os.path.join(out_root, f"mix_stats{stats_suffix}.json")
    write_json(stats_path, mix_stats)

    print(f"Saved mixed clean: {out_clean_path}")
    print(f"Saved mixed dirty: {out_dirty_path}")
    print(f"Saved mix stats: {stats_path}")


if __name__ == "__main__":
    main()
