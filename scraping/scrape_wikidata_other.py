import requests
import re
import mwparserfromhell
from transformers import AutoTokenizer
import os
import json

TUTORIAL_PAGES = [
    "Wikidata:SPARQL_tutorial",
    "Wikidata:List_of_properties/Query_examples",
]

tokenizer = AutoTokenizer.from_pretrained("t5-base")  # or your model

OUT_FILE = "wikidata_tutorial_examples_sparql.jsonl"
CHECKPOINT_FILE = "wikidata_tutorial_checkpoint.json"

def fetch_wikitext(title):
    resp = requests.get("https://www.wikidata.org/w/api.php", {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "titles": title,
        "format": "json",
        "formatversion": 2,
    })
    pages = resp.json()["query"]["pages"]
    try:
        return pages[0]["revisions"][0]["content"]
    except:
        print(f"Failed to fetch wikitext for {title}")
        return ""

def clean_wiki_markup(text):
    code = mwparserfromhell.parse(text)
    return code.strip_code()

def extract_sparql_blocks_and_context(wikitext):
    """
    Finds all <syntaxhighlight lang="sparql">...</syntaxhighlight> blocks,
    grabs surrounding lines as context (e.g., previous 1-2 paragraphs or section header).
    """
    # Find all code blocks
    pattern = re.compile(r'<syntaxhighlight lang="sparql">(.*?)</syntaxhighlight>', re.DOTALL)
    results = []
    for match in pattern.finditer(wikitext):
        query = match.group(1).strip()
        # Get text before the code block for context (look for previous section/title or paragraph)
        before = wikitext[:match.start()]
        # Try to get last heading or non-empty paragraph above
        prev_sections = re.split(r"(={2,}.*?={2,})", before)
        context = ""
        if len(prev_sections) > 2:
            # Last section header + whatever follows
            context = prev_sections[-2] + prev_sections[-1]
        else:
            # Otherwise just the last few lines/paragraphs
            context = "\n".join(before.splitlines()[-8:])
        context_clean = clean_wiki_markup(context)
        results.append({"context": context_clean, "query": query})
    return results

def create_token_blocks(context, query, tokenizer, target_tokens=512):
    text_block = f"{context}\n\nSPARQL Query:\n{query}"
    tokens = tokenizer(text_block)["input_ids"]
    if len(tokens) > target_tokens:
        context_tokens = tokenizer(context)["input_ids"]
        query_tokens = tokenizer(query)["input_ids"]
        keep_context = max(0, target_tokens - len(query_tokens) - 10)
        context_trimmed = tokenizer.decode(context_tokens[:keep_context])
        text_block = f"{context_trimmed}\n\nSPARQL Query:\n{query}"
    return text_block

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(all_samples):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(all_samples), f, ensure_ascii=False)

def main():
    all_samples = load_checkpoint()
    print(f"Loaded {len(all_samples)} keys from checkpoint.")

    processed_this_run = 0

    for title in TUTORIAL_PAGES:
        print(f"Processing {title} ...")
        wikitext = fetch_wikitext(title)
        if not wikitext:
            continue
        samples = extract_sparql_blocks_and_context(wikitext)
        print(f"  Found {len(samples)} code blocks.")
        for s in samples:
            context = s["context"]
            query = s["query"]
            key = f"{query.strip()}||{context.strip()}"
            if key not in all_samples:
                text_block = create_token_blocks(context, query, tokenizer)
                sample = {
                    "context": context,
                    "query": query,
                    "full_text": text_block,
                    "source": title
                }
                # Incremental append
                with open(OUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                all_samples.add(key)
                processed_this_run += 1
                if processed_this_run % 10 == 0:
                    save_checkpoint(all_samples)
        save_checkpoint(all_samples)

    print(f"Script done. {processed_this_run} new samples processed this run.")
    print(f"Total unique samples so far: {len(all_samples)}")
    print(f"Data saved to {OUT_FILE}. Checkpoint saved to {CHECKPOINT_FILE}.")

if __name__ == "__main__":
    main()
