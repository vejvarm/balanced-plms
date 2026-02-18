from collections import defaultdict
import pathlib
import json
import os
import random

# 1. Load all templates
root = pathlib.Path(__file__).parent
template_dirs = ["v2.0_init",  "v2.1_init"]

all_templates = defaultdict(dict)

for template_dir in template_dirs:
    template_files = root.joinpath(template_dir).glob('*.json')
    for path in template_files:
        if "!data_info" in path.name:
            continue
        filename = os.path.basename(path)
        with open(path, 'r', encoding='utf-8') as f:
            templates = json.load(f)
            all_templates[template_dir][filename] = templates

lsq_root = pathlib.Path("../v1/LSQv2")
lsq_files = lsq_root.glob('LSQv2_*.jsonl')

for lsq_input_path in lsq_files:
    lsq_output_path = lsq_input_path.with_name(lsq_input_path.stem + '_with_context.jsonl')
    with open(lsq_input_path, 'r', encoding='utf-8') as infile, \
        open(lsq_output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            row = json.loads(line)
            # 3. Pick a random template file and template
            template_dir = random.choice(list(all_templates.keys()))
            template_file = random.choice(list(all_templates[template_dir].keys()))
            template_str = random.choice(all_templates[template_dir][template_file])
            # 4. Fill in the template
            query_text = row['query']
            prompt = template_str.replace('{query}', query_text)
            # 5. Add context and template fields
            row['context'] = prompt
            row['template'] = f"{template_dir}/{template_file}"
            # 6. Write to output
            outfile.write(json.dumps(row, ensure_ascii=False) + '\n')
