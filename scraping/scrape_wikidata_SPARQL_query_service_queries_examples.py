from bs4 import BeautifulSoup
import requests, json
from tqdm import tqdm

PAGE = "https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples"

def scrape_examples():
    resp = requests.get(PAGE)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for h3 in tqdm(soup.find_all("h3")):
        title = h3.get_text().strip()
        # Grab next <p> for description
        p = h3.find_next_sibling("p")
        ctx = p.get_text().strip() if p else title

        code_block = h3.find_next("pre")
        if code_block:
            query = code_block.get_text().strip()
            if query and len(query) > 30:
                results.append({
                    "query": query,
                    "context": f"{title}: {ctx}"
                })
    return results

examples = scrape_examples()
print(f"Scraped {len(examples)} examples.")
with open("wikidata_SPARQL_query_service_queries_examples.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
