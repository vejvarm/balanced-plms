import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

def scrape_wikidata_tutorial():
    url = "https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    examples = []
    for header in soup.select("h2, h3"):
        section = header.get_text().strip()
        pre = header.find_next("pre")
        if not pre:
            continue
        query = pre.get_text().strip()
        # Grab explanatory context from preceding paragraphs until next pre
        ctx_parts = []
        for sib in header.find_next_siblings():
            if sib == pre:
                break
            if sib.name in ("p", "ul", "ol"):
                ctx_parts.append(sib.get_text().strip())
        context = section + "\n" + "\n".join(ctx_parts)
        examples.append({"query": query, "context": context})
    return examples

def scrape_wikibooks_sparql(start_url="https://en.wikibooks.org/wiki/SPARQL"):
    to_visit = {start_url}
    seen = set()
    examples = []
    while to_visit:
        url = to_visit.pop()
        seen.add(url)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Collect child links under same domain, exclude non-content
        for a in soup.select("#bodyContent a[href^='/wiki/SPARQL']"):
            href = a["href"]
            full = "https://en.wikibooks.org" + href
            if full not in seen:
                to_visit.add(full)
        # Extract pre blocks with context
        for pre in soup.find_all("pre"):
            query = pre.get_text().strip()
            if not query.lower().startswith(("select", "ask", "construct", "describe")):
                continue
            # Context from nearest preceding <p> or heading
            ctx = ""
            prev = pre.find_previous(lambda t: t.name in ("p", "h2", "h3", "h4"))
            if prev:
                ctx = prev.get_text().strip()
            else:
                ctx = soup.title.get_text()
            examples.append({"query": query, "context": ctx})
    return examples

# Run scrapers
wikidata_examples = scrape_wikidata_tutorial()
wikibooks_examples = scrape_wikibooks_sparql()

print(f"Collected {len(wikidata_examples)} examples from Wikidata tutorial.")
print(f"Collected {len(wikibooks_examples)} examples from Wikibooks SPARQL.")

# Save combined dataset
with open("wikidata_wikibooks_sparql.jsonl", "w", encoding="utf-8") as fout:
    for ex in wikidata_examples + wikibooks_examples:
        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
