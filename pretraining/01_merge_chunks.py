import argparse
import os
import pathlib
from datasets import load_from_disk, concatenate_datasets
from data_utils import load_config


def main(args):
    config_args = load_config(args.ds)
    ds_path = config_args.get("dataset_cache_path", f"./{args.ds}-preproc-chunks")

    chunk_dirs = sorted(
        [os.path.join(ds_path, d, "clean") for d in os.listdir(ds_path) if d.startswith("chunk_")]
    )
    all_chunks = [load_from_disk(d) for d in chunk_dirs]
    full_dataset = concatenate_datasets(all_chunks)
    full_dataset.save_to_disk("openwebtext-preproc-final/clean")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("ds", type=str, choices=("openwebtext", "openwebtext-10k", "realnewslike"))
    main(parser.parse_args())