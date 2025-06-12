import json
from tqdm import tqdm

input_file = "sparql_explain_data.jsonl"
batch_file = ".openai_batch_input.jsonl"

BATCH_MODEL = "gpt-4.1-mini-2025-04-14"
TEMPLATE = open(".template.prompt", "r").read()

def build_prompt(query, context):
    return TEMPLATE.format(query=query, context=context)

with open(input_file, "r", encoding="utf-8") as fin, \
     open(batch_file, "w", encoding="utf-8") as fout:
    for i, line in enumerate(tqdm(fin)):
        d = json.loads(line)
        # Only submit unlabeled
        if d.get("target"):
            continue
        prompt = build_prompt(d["query"], d["context"])
        entry = {
            "custom_id": str(i),  # or use another unique id if available
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": BATCH_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.4
            }
        }
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Batch input file ready: {batch_file}")
