import json
import pathlib

BATCH_RESULT_FOLDER = pathlib.Path("./.batch_results")
INPUT_FILE = "sparql_explain_data.jsonl"
OUTPUT_FILE = "sparql_explain_data_labeled.jsonl"

# 1. Collect all batch result files
batch_files = sorted(BATCH_RESULT_FOLDER.glob("openai_batch_output_*.jsonl"))

# 2. Build a mapping: custom_id -> explanation
custom_id_to_content = {}

for batch_file in batch_files:
    print(f"Reading {batch_file} ...")
    with open(batch_file, "r", encoding="utf-8") as fin:
        for line in fin:
            d = json.loads(line)
            custom_id = d.get("custom_id")
            try:
                content = d["response"]["body"]["choices"][0]["message"]["content"].strip()
            except Exception as e:
                content = ""
            if custom_id:  # Avoid accidental overwriting
                custom_id_to_content[custom_id] = content

print(f"Total unique completions found: {len(custom_id_to_content)}")

# 3. Merge with your original prompts
with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        d = json.loads(line)
        custom_id = str(i)
        if custom_id in custom_id_to_content and not d.get("target"):
            d["target"] = custom_id_to_content[custom_id]
        fout.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"Final merged file written to {OUTPUT_FILE}")
