import argparse
import json
import os
import shutil
from datetime import datetime, timezone

from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge grouped chunk datasets created by 06_stream_c4_noclean_match_owt.py."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root output dir from 06_stream_c4_noclean_match_owt.py (contains grouped_chunks/).",
    )
    parser.add_argument("--chunks_subdir", type=str, default="grouped_chunks")
    parser.add_argument("--final_name", type=str, default="final_clean_grouped")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument(
        "--overwrite_final",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing final dataset path if present.",
    )
    parser.add_argument(
        "--shared_dev_from",
        type=str,
        default=None,
        help="Path to an existing grouped dev dataset to reuse (typically OWT shared_dev_grouped/grouped).",
    )
    parser.add_argument(
        "--shared_dev_mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="How to place shared dev set under <input_dir>/shared_dev_grouped/grouped.",
    )
    return parser.parse_args()


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()

    chunks_dir = os.path.join(args.input_dir, args.chunks_subdir)
    if not os.path.isdir(chunks_dir):
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    chunk_paths = sorted(
        [
            os.path.join(chunks_dir, d)
            for d in os.listdir(chunks_dir)
            if d.startswith("chunk_") and os.path.isdir(os.path.join(chunks_dir, d))
        ]
    )
    if not chunk_paths:
        raise ValueError(f"No chunk_* datasets found under: {chunks_dir}")

    final_path = os.path.join(args.input_dir, args.final_name)
    if os.path.exists(final_path):
        if args.overwrite_final:
            shutil.rmtree(final_path)
        else:
            raise FileExistsError(
                f"Final dataset already exists: {final_path}. Use --overwrite_final to replace."
            )

    print(f"Found {len(chunk_paths)} chunk datasets. Loading and concatenating...")
    datasets = []
    for chunk_path in tqdm(chunk_paths, desc="Load chunks", ncols=100):
        datasets.append(load_from_disk(chunk_path))

    merged = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    print(f"Merged blocks: {len(merged)}")
    print(f"Saving final merged dataset to: {final_path}")
    merged.save_to_disk(final_path)

    num_blocks = len(merged)
    num_tokens = int(num_blocks * args.block_size)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": args.input_dir,
        "chunks_dir": chunks_dir,
        "n_chunks": len(chunk_paths),
        "final_dataset_path": final_path,
        "num_blocks": int(num_blocks),
        "num_tokens_assuming_fixed_block_size": num_tokens,
        "block_size": args.block_size,
        "column_names": merged.column_names,
    }

    summary_path = os.path.join(args.input_dir, "merge_summary.json")
    write_json(summary_path, summary)
    print(f"Saved merge summary: {summary_path}")

    final_stats_path = os.path.join(args.input_dir, "final_stats.json")
    write_json(
        final_stats_path,
        {
            "final_clean": {
                "num_blocks": int(num_blocks),
                "num_tokens": int(num_tokens),
            }
        },
    )
    print(f"Saved final stats: {final_stats_path}")

    if args.shared_dev_from:
        if not os.path.exists(args.shared_dev_from):
            raise FileNotFoundError(f"shared_dev_from path does not exist: {args.shared_dev_from}")

        shared_dev_group_dir = os.path.join(args.input_dir, "shared_dev_grouped")
        os.makedirs(shared_dev_group_dir, exist_ok=True)
        shared_dev_target = os.path.join(shared_dev_group_dir, "grouped")
        if os.path.lexists(shared_dev_target):
            if os.path.islink(shared_dev_target) or os.path.isfile(shared_dev_target):
                os.unlink(shared_dev_target)
            else:
                shutil.rmtree(shared_dev_target)

        if args.shared_dev_mode == "copy":
            print(f"Copying shared dev grouped dataset to: {shared_dev_target}")
            shutil.copytree(args.shared_dev_from, shared_dev_target)
        else:
            print(f"Symlinking shared dev grouped dataset to: {shared_dev_target}")
            os.symlink(args.shared_dev_from, shared_dev_target)

    print("Done.")


if __name__ == "__main__":
    main()
