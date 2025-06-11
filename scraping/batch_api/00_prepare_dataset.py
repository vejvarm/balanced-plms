import json
import glob
from tqdm import tqdm

# List your source files
input_files = [
    "../wikidata_weekly_examples_sparql.jsonl",
    "../wikidata_tutorial_examples_sparql.jsonl",
    # add other sources as needed
]

output_file = "sparql_explain_data.jsonl"
n_written = 0

with open(output_file, "w", encoding="utf-8") as fout:
    for fname in input_files:
        with open(fname, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=fname):
                d = json.loads(line)
                # Use query and context; you can add more prompt engineering here if desired
                # prompt = f"Explain this SPARQL query:\n{d['query']}\nContext: {d['context']}"
                # Use human annotation if present, otherwise leave blank for semi-supervised
                target = d.get("target", "")
                sample = {"query": d['query'], "context": d['context'], "target": target}
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written += 1

print(f"Wrote {n_written} examples to {output_file}")
