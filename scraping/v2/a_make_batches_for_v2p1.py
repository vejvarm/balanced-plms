import json
import random
import os
from tqdm import tqdm

TEMPLATE_DIR = "v2.0_init"
BATCH_DIR = "v2.1_expand/.batches"
os.makedirs(BATCH_DIR, exist_ok=True)

DATA_INFO_PATH = os.path.join(TEMPLATE_DIR, "!data_info.json")

MODELS = ["gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"]
PROMPT_TEMPLATES_PATH = "v2.1_expand/prompt_templates.json"
TEMPS = [0.25, 0.5, 0.75, 1.0]
NUM_EXAMPLES = 20
NUM_COMPLETIONS = 300
NUM_RUNS = 1

# Load data_info.json and prompt template
with open(DATA_INFO_PATH, "r", encoding="utf-8") as f:
    DATA_INFO = json.load(f)

with open(PROMPT_TEMPLATES_PATH, "r", encoding="utf-8") as f:
    PROMPT_VAR_LIST = json.load(f)

template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(".json")]
PROMPT_VARIATIONS = [1]  # exlucing 0, which we already used 

for model in MODELS:
    out_file = os.path.join(BATCH_DIR, f"openai_batch_input_{model}.jsonl")
    with open(out_file, "w", encoding="utf-8") as fout:
        for prompt_var in PROMPT_VARIATIONS:
            prompt_template = PROMPT_VAR_LIST[prompt_var]
            print(f"Creating batch for model: {model}, prompt variation: {prompt_var}...")
            global_idx = 0
            for template_file in template_files:
                template_path = os.path.join(TEMPLATE_DIR, template_file)
                print(f"Processing {template_file} for {model}...")

                with open(template_path, "r", encoding="utf-8") as f:
                    templates = json.load(f)

                info = DATA_INFO.get(template_file, None)
                if not info:
                    print(f"Metadata for {template_file} not found in !data_info.json, skipping.")
                    continue

                for temp in TEMPS:
                    for run_num in range(NUM_RUNS):
                        examples = random.sample(templates, k=min(NUM_EXAMPLES, len(templates)))
                        examples_str = json.dumps(examples, ensure_ascii=False, indent=2)
                        prompt = prompt_template.format(
                            description=info.get("description", ""),
                            tags=", ".join(info.get("tags", [])),
                            examples=examples_str,
                            N=NUM_COMPLETIONS
                        )
                        entry = {
                            "custom_id": (
                                f"{template_file}_{model}_t{str(temp).replace('.','')}_p{prompt_var}_"
                                f"run{run_num}_idx{global_idx}"
                            ),
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 5000,
                                "temperature": temp,
                            },
                        }
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        global_idx += 1

    print(f"Batch file written: {out_file}")
print("All batches created successfully!")