import json
import re
from datasets import load_dataset
from tqdm import tqdm

# --- Estimated document counts for C4 variants ---
C4_LENGTHS = {
    "en": 365_000_000,            # 365 million docs
    "en.noblocklist": 395_000_000,# 395 million docs
    "en.noclean": 1_100_000_000   # 1.1 billion docs
}

# --- Filtering logic for SPARQL-like text ---

# SQL_CYPHER = [
#     "join", "merge", "match", "return", "create", "insert", "delete"
# ]
# SQL_CYPHER_RE = re.compile(r"\b(" + "|".join(SQL_CYPHER) + r")\b")

FORM_KEYWORDS_RE = re.compile(r"\b(select|ask|construct|describe)\b")
PREFIX_RE = re.compile(r"prefix\s+[^\s:]+:\s*<[^>]+>", re.I)
TRIPLE_PATTERN_RE = re.compile(r"\?\w+\s+\S+\s+\?\w+\s*\.")

def looks_like_sparql(sample):
    text = sample["text"].lower()
    # if "sparql" in text and not SQL_CYPHER_RE.search(text):
    #     return True
    
    if not (FORM_KEYWORDS_RE.search(text) or PREFIX_RE.search(text)):
        return False
    if "{" not in text or "}" not in text:
        return False
    contents = re.findall(r"\{(.*?)\}", text, re.DOTALL)
    if not any(TRIPLE_PATTERN_RE.search(c) for c in contents):
        return False
    # if SQL_CYPHER_RE.search(text):
    #     return False
    return True

# --- Stream C4, filter, and save incrementally ---

def main(
    outfile="c4_sparql.jsonl",
    max_examples=None,
    variant="en.noclean"
):
    print(f"Loading C4 dataset stream with variant: {variant} ...")
    ds = load_dataset("allenai/c4", variant, split="train", streaming=True)
    total = C4_LENGTHS.get(variant, None)
    print(f"Estimated documents in split: {total if total else 'Unknown'}")

    saved = 0
    with tqdm(total=total, unit="docs", desc="Scanning C4", dynamic_ncols=True) as pbar:
        for sample in ds:
            pbar.update(1)
            if looks_like_sparql(sample):
                with open(outfile, "a", encoding="utf-8") as fout:
                    json.dump(sample, fout, ensure_ascii=False)
                    fout.write("\n")
                saved += 1
                pbar.set_postfix(saved=saved)
                # if print_every and saved % print_every == 0:
                #     print(f"Saved {saved} SPARQL-like samples so far")
            if max_examples and saved >= max_examples:
                print("Reached max_examples cap, exiting.")
                break
    print(f"Finished! Total saved: {saved}")

if __name__ == "__main__":
    main()
