import json
from tqdm import tqdm

input_files = [
    "../stackexchange_sparql_rdf_examples.jsonl",
    # add any other Stack Exchange dumps here
]
output_file = "sparql_stackexchange_data.jsonl"
n_written = 0

with open(output_file, "w", encoding="utf-8") as fout:
    for fname in input_files:
        with open(fname, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=fname):
                d = json.loads(line)
                sample = {
                    "query": d["query"],
                    "context": d["context"],
                    "post_type": d.get("post_type", "question"),  # Always keep this for prompt injection
                    "target": d.get("target", "")
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written += 1

print(f"Wrote {n_written} examples to {output_file}")
