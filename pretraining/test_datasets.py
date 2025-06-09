import json
import sys
import os
from datasets import load_from_disk
from transformers import AutoTokenizer

def get_config_path(ds_name):
    mapping = {
        "openwebtext": "./configs/00_config_openwebtext.json",
        "openwebtext-10k": "./configs/00_config_openwebtext-10k.json",
        "realnewslike": "./configs/00_config_c4-realnewslike.json"
    }
    if ds_name not in mapping:
        raise ValueError(f"Unknown dataset: {ds_name}. Choose from: {list(mapping.keys())}")
    return mapping[ds_name]

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_check_grouped_samples.py [openwebtext|openwebtext-10k|realnewslike]")
        sys.exit(1)
    ds_name = sys.argv[1]
    cfg_path = get_config_path(ds_name)
    with open(cfg_path, "r") as f:
        config_args = json.load(f)
    cache_path = config_args.get("dataset_cache_path")
    model_name = config_args["model_name_or_path"]
    if not cache_path:
        print("No 'dataset_cache_path' in config.")
        sys.exit(1)

    clean_path = os.path.join(cache_path, "final_clean_grouped")
    dirty_path = os.path.join(cache_path, "final_dirtymatched_grouped")
    print(f"Loading final_clean_grouped from: {clean_path}")
    clean = load_from_disk(clean_path)
    print(f"Loading final_dirtymatched_grouped from: {dirty_path}")
    dirty = load_from_disk(dirty_path)
    print(f"\nLoaded datasets. Now printing some detokenized samples...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def print_samples(ds, label):
        print(f"\n[{label}] Total blocks: {len(ds)}")
        for i in range(min(3, len(ds))):
            sample = ds[i]
            input_ids = sample.get("input_ids", None)
            if input_ids is None:
                print(f"Sample {i+1} missing 'input_ids' field, full sample: {sample}")
                continue
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"\n{label} sample {i+1}:")
            print(f"  input_ids: {input_ids[:10]}{'...' if len(input_ids) > 10 else ''} (len={len(input_ids)})")
            print(f"  text: {repr(text)}")

    print_samples(clean, "Clean")
    print_samples(dirty, "DirtyMatched")

if __name__ == "__main__":
    main()
