import json
from tqdm import tqdm
import random
import os

input_file = "sparql_stackexchange_data.filtered.jsonl"
batch_dir = ".batches"
os.makedirs(batch_dir, exist_ok=True)

MODELS = ["gpt-4.1-mini-2025-04-14", "gpt-4o-mini-2024-07-18"]
TEMPS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
PROMPTS = json.load(open(".stack.template.prompts.json"))

# Load data once
samples = []
with open(input_file, "r", encoding="utf-8") as fin:
    for line in fin:
        d = json.loads(line)
        samples.append(d)

for model in MODELS:
    for temp in TEMPS:
        batch_file = os.path.join(
            batch_dir, f"openai_batch_input_{model}_t{str(temp).replace('.','')}.jsonl"
        )
        with open(batch_file, "w", encoding="utf-8") as fout:
            for i, d in enumerate(tqdm(samples, desc=f"{model} t={temp}")):
                if d.get("target"):
                    continue
                prompt_template = random.choice(PROMPTS)
                prompt = prompt_template.format(
                    query=d["query"], context=d["context"], post_type=d.get("post_type", "question")
                )
                entry = {
                    "custom_id": f"{i}_t{str(temp).replace('.','')}_{PROMPTS.index(prompt_template)}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 250,
                        "temperature": temp
                    },
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Batch input file ready: {batch_file}")
