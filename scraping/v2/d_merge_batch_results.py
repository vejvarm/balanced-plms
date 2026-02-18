import os
import json
import glob

BATCH_RESULTS_DIR = "v2.1_expand/.batch_results"
OUTPUT_DIR = "v2.1_expand"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def patch_json_list(json_str):
    json_str = json_str.rstrip(" \n")
    if json_str.endswith(']'):
        return json_str
    if not json_str.endswith('"'):
        json_str += '"'
    return json_str.rstrip(' "\n,') + '"\n]'

def parse_custom_id(custom_id):
    # expects format: {template_file}_{model}_tXX_runN_idxN
    # example: templates-explain-q.json_gpt-4.1-nano-2025-04-14_t05_run0_idx93
    parts = custom_id.split("_")
    template_file = parts[0]
    model = parts[1]
    return template_file, model

# Main dict for merging: {(template, model): [rows]}
merged = {}

file_list = glob.glob(os.path.join(BATCH_RESULTS_DIR, "openai_batch_output_*.jsonl"))
num_written = 0
num_skipped = 0

for batch_path in file_list:
    print(f"Processing {batch_path}")
    with open(batch_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            try:
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if not custom_id or "response" not in row:
                    num_skipped += 1
                    continue
                content = row["response"]["body"]["choices"][0]["message"]["content"]
                start = content.find('[')
                if start == -1:
                    print(f"Could not find json list in response for {custom_id} (line {line_num})")
                    num_skipped += 1
                    continue
                json_str = content[start:].rstrip(" \n")
                if not json_str.endswith(']'):
                    json_str = patch_json_list(json_str)
                try:
                    data = json.loads(json_str)
                except Exception as e:
                    print(f"JSON decode error in {custom_id} (line {line_num}): {e}")
                    num_skipped += 1
                    continue
                data_clean = [row for row in data if "{query}" in row]
                template_file, model = parse_custom_id(custom_id)
                key = (template_file, model)
                merged.setdefault(key, []).extend(data_clean)
                num_written += 1
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                num_skipped += 1

# Write merged files
for (template_file, model), all_rows in merged.items():
    # Remove duplicates while preserving order
    seen = set()
    out_rows = []
    for row in all_rows:
        if row not in seen:
            out_rows.append(row)
            seen.add(row)
    # Make output file name
    template_stub = template_file.replace(".json", "")
    fname = f"final_{template_stub}_{model}.json"
    outpath = os.path.join(OUTPUT_DIR, fname)
    with open(outpath, "w", encoding="utf-8") as fout:
        json.dump(out_rows, fout, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out_rows)} prompts to {fname}")

print(f"\nDone. {num_written} batch entries processed, {num_skipped} skipped.")
