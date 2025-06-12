import requests
import time
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

OUT_FILE = "stackexchange_sparql_rdf_examples.jsonl"
PAGE_SIZE = 100  # max for API
MAX_PAGES = 100  # change for more data (API is rate-limited)
STACK_API_KEY = "rl_iTx9ydkTuoeypDDbkmzqoEaF1"  # add your API key if you have one for more quota

def has_forbidden_tag(tags):
    forbidden = {"neo4j", "sql", "cypher"}
    return any(tag in forbidden for tag in tags)

def has_required_tag(tags):
    required = {"sparql", "rdf"}
    return any(tag in required for tag in tags)

def extract_code_blocks(html):
    soup = BeautifulSoup(html, "html.parser")
    return [c.get_text() for c in soup.find_all("code")]

def remove_code_blocks(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("code"):
        tag.decompose()
    return soup.get_text(separator="\n").strip()

def rich_context_for_question(title, body, code):
    """Combine question title and full body (minus code) as context."""
    context = f"{title.strip()}\n{remove_code_blocks(body)}"
    return context.strip()

def rich_context_for_answer(ans_body, question_title, question_body, code):
    """Combine question title/body and answer body (all minus code) as context."""
    question_part = f"{question_title.strip()}\n{remove_code_blocks(question_body)}"
    answer_part = remove_code_blocks(ans_body)
    context = f"{question_part}\n\n{answer_part}"
    return context.strip()

def fetch_questions(page):
    params = {
        "order": "desc",
        "sort": "votes",
        "tagged": "sparql;rdf",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": PAGE_SIZE,
        "page": page,
    }
    if STACK_API_KEY:
        params["key"] = STACK_API_KEY
    resp = requests.get("https://api.stackexchange.com/2.3/questions", params=params)
    resp.raise_for_status()
    return resp.json()

def fetch_answers(question_id):
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": PAGE_SIZE,
    }
    if STACK_API_KEY:
        params["key"] = STACK_API_KEY
    resp = requests.get(f"https://api.stackexchange.com/2.3/questions/{question_id}/answers", params=params)
    resp.raise_for_status()
    return resp.json()

def is_sparql_like(code):
    text = code.lower()
    return "select" in text or "ask" in text or "prefix" in text or "where" in text

n_written = 0
with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for page in tqdm(range(1, MAX_PAGES + 1)):
        try:
            data = fetch_questions(page)
        except Exception as e:
            print(f"Error fetching questions on page {page}: {e}")
            time.sleep(10)
            continue

        for q in data.get("items", []):
            tags = set(q["tags"])
            if not has_required_tag(tags) or has_forbidden_tag(tags):
                continue

            question_title = q.get("title", "")
            question_body = q.get("body", "")
            # Extract SPARQL code blocks from question
            question_code_blocks = [c for c in extract_code_blocks(question_body) if is_sparql_like(c)]
            for code in question_code_blocks:
                context = rich_context_for_question(question_title, question_body, code)
                record = {
                    "source": f"https://stackoverflow.com/questions/{q['question_id']}",
                    "query": code.strip(),
                    "context": context,
                    "post_type": "question"
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            # Now, fetch and process answers
            try:
                answers = fetch_answers(q["question_id"])
            except Exception as e:
                print(f"Error fetching answers for QID {q['question_id']}: {e}")
                time.sleep(5)
                continue

            for ans in answers.get("items", []):
                ans_body = ans.get("body", "")
                ans_code_blocks = [c for c in extract_code_blocks(ans_body) if is_sparql_like(c)]
                for code in ans_code_blocks:
                    context = rich_context_for_answer(ans_body, question_title, question_body, code)
                    record = {
                        "source": f"https://stackoverflow.com/a/{ans['answer_id']}",
                        "query": code.strip(),
                        "context": context,
                        "post_type": "answer"
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
        # Respect StackExchange API rate limits!
        time.sleep(1)
        if not data.get("has_more", False):
            break

print(f"Done. Wrote {n_written} SPARQL/RDF examples with rich context to {OUT_FILE}")
