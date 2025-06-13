import glob
import json
import os
from collections import defaultdict

batch_dir = ".batch_results"
subdir = "stackexchange"
input_file = "sparql_stackexchange_data.filtered.jsonl"
output_file = "stackexchange_paraphrases_multi_temp.jsonl"

input_data = []
with open(input_file, "r", encoding="utf-8") as fin:
    for line in fin:
        input_data.append(json.loads(line))

merged = defaultdict(list)
batch_files = sorted(glob.glob(f"{batch_dir}/{subdir}/openai_batch_output_*.jsonl"))

total_count = len(input_data)
for batch_file in batch_files:
    print(f"Processing: {batch_file}")
    with open(batch_file, "r", encoding="utf-8") as fin:
        for line in fin:
            d = json.loads(line)
            cid = d.get("custom_id")
            try:
                # Parse: "{idx}_t{temp_no_dot}_{promptidx}"
                parts = cid.split("_")
                idx = int(parts[0])
                # Get temp from tXX (e.g., t15 → 0.15)
                temp_str = parts[1][1:]  # remove 't'
                temp = float(f"0.{temp_str}")
                prompt_template_index = int(parts[2]) if len(parts) > 2 else None
                # Model comes from the response body!
                model = d["response"]["body"].get("model", None)
                content = d["response"]["body"]["choices"][0]["message"]["content"].strip()
                merged[idx].append({
                    "paraphrase": content,
                    "model": model,
                    "temperature": temp,
                    "custom_id": cid,
                    "batch_file": os.path.basename(batch_file),
                    "prompt_template_index": prompt_template_index
                })
                total_count += 1
            except Exception as e:
                print(f"Skipping malformed line: {e}")

# Write merged file
with open(output_file, "w", encoding="utf-8") as fout:
    for idx, d in enumerate(input_data):
        all_paraphrases = merged.get(idx, [])
        out = {
            "query": d["query"],
            "context": d["context"],
            "post_type": d.get("post_type", "question"),
            "paraphrases": all_paraphrases
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"Wrote merged paraphrases to {output_file}. The total context sample count is {total_count}.")
