import requests
import re
from urllib.parse import unquote
import mwparserfromhell
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import os
import json

WIKIDATA_PAGES = [
    "Wikidata:Weekly_query_examples",
    "Wikidata:Weekly_query_examples/2019",
    "Wikidata:Weekly_query_examples/2020",
    "Wikidata:Weekly_query_examples/2021",
    "Wikidata:Weekly_query_examples/2022",
    "Wikidata:Weekly_query_examples/2023",
    "Wikidata:Weekly_query_examples/2024",
    "Wikidata:Weekly_query_examples/2025"
]

tokenizer = AutoTokenizer.from_pretrained("t5-base")  # or your actual model

OUT_FILE = "wikidata_weekly_examples_sparql.jsonl"
CHECKPOINT_FILE = "wikidata_checkpoint.json"

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

def extract_shortlinks_and_context(wikitext):
    pattern = re.compile(r'\*\s*\[(https://w\.wiki/\w+)\s+([^\]]+)\]')
    matches = pattern.findall(wikitext)
    return [(desc.strip(), shortlink.strip()) for shortlink, desc in matches]

def expand_and_get_sparql(short_link):
    try:
        resp = requests.get(short_link, allow_redirects=True, timeout=10)
        url = resp.url
        if '#' in url:
            sparql_encoded = url.split('#', 1)[-1]
            return unquote(sparql_encoded)
    except Exception as e:
        print(f"Failed to expand {short_link}: {e}")
    return None

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

    for title in WIKIDATA_PAGES:
        print(f"Processing {title} ...")
        wikitext = fetch_wikitext(title)
        if not wikitext:
            continue
        shortlinks = extract_shortlinks_and_context(wikitext)
        print(f"  Found {len(shortlinks)} candidate queries.")
        for context, link in tqdm(shortlinks):
            context_clean = clean_wiki_markup(context)
            query = expand_and_get_sparql(link)
            if query:
                key = f"{query.strip()}||{context_clean.strip()}"
                if key not in all_samples:
                    text_block = create_token_blocks(context_clean, query, tokenizer)
                    sample = {
                        "context": context_clean,
                        "query": query,
                        "full_text": text_block,
                        "source": link
                    }
                    # Incremental append
                    with open(OUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    all_samples.add(key)
                    processed_this_run += 1
                    # Save checkpoint every 10 queries processed
                    if processed_this_run % 10 == 0:
                        save_checkpoint(all_samples)
                    # Be polite to Wikidata: 1-2s delay
                    time.sleep(.3)
            else:
                print(f"  Query not found for: {link}")

    save_checkpoint(all_samples)
    print(f"Script done. {processed_this_run} new samples processed this run.")
    print(f"Total unique samples so far: {len(all_samples)}")
    print(f"Data saved to {OUT_FILE}. Checkpoint saved to {CHECKPOINT_FILE}.")

if __name__ == "__main__":
    main()
