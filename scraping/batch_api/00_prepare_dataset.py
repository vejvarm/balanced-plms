import json
import glob
from tqdm import tqdm

# List your source files
# BASE
# input_files = [
#     "../wikidata_weekly_examples_sparql.jsonl",
#     # add other sources as needed
# ]


"""
input_files1 = [
    # "../wikidata_weekly_examples_sparql.jsonl",
]

input_files2 = [
    "../wikidata_SPARQL_query_service_queries_examples.jsonl",
    "../wikidata_sparql_tutorial_examples.jsonl",
    "../wikidata_wikibase_rdf_format_examples.jsonl",
    "../wikidata_wikibooks_sparql.jsonl"
    # add other sources as needed
]
"""

# EXTRA
input_files = [
    "../wikidata_SPARQL_query_service_queries_examples.jsonl",
    "../wikidata_sparql_tutorial_examples.jsonl",
    "../wikidata_weekly_examples_sparql.jsonl",
    "../wikidata_wikibase_rdf_format_examples.jsonl",
    "../wikidata_wikibooks_sparql.jsonl",
    "../v1/LSQv2/part1_stratified/LSQv2_Easy.jsonl",
    "../v1/LSQv2/part1_stratified/LSQv2_Medium.jsonl",
    "../v1/LSQv2/part1_stratified/LSQv2_Hard.jsonl",
    "../v1/LSQv2/part1_stratified/LSQv2_ExtraHard.jsonl",
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
                sample = {"query": d.get("query"), "context": d.get('context', ""), "target": target}
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written += 1

print(f"Wrote {n_written} examples to {output_file}")
