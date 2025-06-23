import argparse
import json
import os
from datasets import load_from_disk, concatenate_datasets
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inject SPARQL explanation blocks into OpenWebText")
    parser.add_argument("--owt_dir", default="/work/datasets/openwebtext-preproc", help="Path to openwebtext preprocessed dataset")
    parser.add_argument("--sparql_dir", default="/work/datasets/sparql_explain_arrow", help="Path to SPARQL Arrow dataset")
    parser.add_argument("--output_dir", default="/work/datasets/openwebtext-injected-SPARQL-10p", help="Output directory")
    parser.add_argument("--proportion", type=float, default=0.1, help="Proportion of OpenWebText blocks to replace")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    owt_path = os.path.join(args.owt_dir, "final_clean_grouped")
    dev_path = os.path.join(args.owt_dir, "shared_dev_grouped", "grouped")
    sparql_path = os.path.join(args.sparql_dir, "grouped")

    print(f"Loading OpenWebText from {owt_path}")
    owt = load_from_disk(owt_path)
    print(f"Loading SPARQL dataset from {sparql_path}")
    sparql = load_from_disk(sparql_path)
    print(f"OWT blocks: {len(owt)}, SPARQL blocks: {len(sparql)}")

    num_replace = int(len(owt) * args.proportion)
    print(f"Replacing {num_replace} blocks ({args.proportion*100:.0f}% of OWT)")

    replace_indices = rng.choice(len(owt), size=num_replace, replace=False)
    keep_indices = np.setdiff1d(np.arange(len(owt)), replace_indices)
    owt_kept = owt.select(keep_indices.tolist())

    if len(sparql) >= num_replace:
        sparql_sample = sparql.shuffle(seed=args.seed).select(range(num_replace))
    else:
        repeats = num_replace // len(sparql)
        remainder = num_replace % len(sparql)
        parts = [sparql] * repeats
        if remainder:
            parts.append(sparql.shuffle(seed=args.seed).select(range(remainder)))
        sparql_sample = concatenate_datasets(parts)

    merged = concatenate_datasets([owt_kept, sparql_sample]).shuffle(seed=args.seed)

    stats = {
        "num_replaced": len(replace_indices),
        "num_kept": len(owt_kept),
        "num_injected": len(sparql_sample),
        "total_after_injection": len(merged)
    }
    print(stats)

    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_to_disk(os.path.join(args.output_dir, "final_clean_grouped"))
    json.dump(stats, open(os.path.join(args.output_dir, "final_stats.json"), "w"), indent=4)
    # copy dev split
    dev = load_from_disk(dev_path)
    dev_out = os.path.join(args.output_dir, "shared_dev_grouped")
    os.makedirs(dev_out, exist_ok=True)
    dev.save_to_disk(os.path.join(dev_out, "grouped"))
    print(f"Saved merged dataset to {args.output_dir}/final_clean_grouped")


if __name__ == "__main__":
    main()
