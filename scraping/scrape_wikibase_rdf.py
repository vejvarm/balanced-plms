import requests
from bs4 import BeautifulSoup
import json

def scrape_wikibase_rdf_dump_format():
    url = "https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format#Data_model"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    all_elements = list(soup.body.descendants)
    last_heading = None
    buffer_context = []
    results = []

    for el in all_elements:
        if getattr(el, 'name', None) in ('h2', 'h3', 'h4'):
            last_heading = el.get_text().strip()
            buffer_context = []
        elif getattr(el, 'name', None) in ('p', 'ul', 'dl'):
            text = el.get_text().strip()
            if text:
                buffer_context.append(text)
        elif getattr(el, 'name', None) == 'pre':
            code_text = el.get_text().strip()
            # Only keep RDF/Turtle/SPARQL blocks (optional: filter on content or parent class)
            if any(x in code_text.lower() for x in ("prefix", "select", "construct", "@prefix", "rdf:", "wd:", "wdt:", "turtle", "sparql")):
                context = " ".join(buffer_context[-3:])
                results.append({
                    "query": code_text,
                    "context": (last_heading or "") + " | " + context,
                    "source": url
                })
    return results

examples = scrape_wikibase_rdf_dump_format()
print(f"Collected {len(examples)} code/context examples from RDF Dump Format.")
with open("wikidata_wikibase_rdf_format_examples.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
