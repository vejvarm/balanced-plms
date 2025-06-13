import requests
from bs4 import BeautifulSoup
import json

def scrape_wikidata_sparql_tutorial():
    url = "https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")

    examples = []
    all_elements = list(soup.body.descendants)
    last_heading = None
    buffer_context = []

    for i, el in enumerate(all_elements):
        if getattr(el, 'name', None) in ('h2', 'h3', 'h4'):
            last_heading = el.get_text().strip()
            buffer_context = []
        elif getattr(el, 'name', None) in ('p', 'ul'):
            # Store context until a code appears
            text = el.get_text().strip()
            if text:
                buffer_context.append(text)
        elif getattr(el, 'name', None) == 'pre':
            code_text = el.get_text().strip()
            # Only take "SELECT"/SPARQL queries
            if any(code_text.lower().startswith(kw) for kw in ("select", "ask", "construct", "describe", "prefix")):
                context = " ".join(buffer_context[-3:])  # up to 3 most recent blocks
                examples.append({
                    "query": code_text,
                    "context": (last_heading or "") + " | " + context
                })
    return examples

examples = scrape_wikidata_sparql_tutorial()
print(f"Collected {len(examples)} examples.")
with open("wikidata_sparql_tutorial_examples.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")