import praw
import re
import time
import json
from tqdm import tqdm

# --- USER: Fill these in! ---
CLIENT_ID = "XTxEyMY7l4xT4TkvvUZ0wg"
CLIENT_SECRET = "b2xB1EBp--pJhy3mKMi0Tq3I1kOYhg"
USER_AGENT = "sparql-scraper/0.1 by YOUR_USERNAME"

SUBREDDITS = [
    "semanticweb",
    "Wikidata",
    "KnowledgeGraph",
    "sparql"
]
SEARCH_TERMS = ["sparql", "PREFIX", "wdt:", "rdf", "construct", "ask"]
MAX_POSTS_PER_SUB = 1000
MAX_COMMENTS_PER_SUB = 2000
MIN_QUERY_LEN = 30

# Unique keywords that mean "definitely SQL/Cypher" and *not* SPARQL
SQL_ONLY = {
    "insert into", "update set", "delete from", "create table", "alter table", "drop table",
    "truncate table", "auto_increment", "foreign key", "primary key", "on duplicate key", "values ("
}
CYPHER_ONLY = {
    "merge ", "unwind ", "detach delete", "set ", "create (", "with ", "return ", "match ("
}

def is_strongly_not_sparql(query, context):
    txt = (query + " " + context).lower()
    # Cypher/SQL are case-insensitive, so always lowercased
    return (
        any(k in txt for k in SQL_ONLY) or
        any(k in txt for k in CYPHER_ONLY)
    )

# --- Connect to Reddit API ---
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
    ratelimit_seconds=60
)

def extract_code_blocks(text):
    # Triple-backtick blocks (Reddit/Markdown)
    blocks = re.findall(r"```(?:sparql)?([\s\S]*?)```", text, re.IGNORECASE)
    # Indented blocks
    indented_blocks = []
    lines = text.split("\n")
    curr = []
    for line in lines:
        if line.startswith("    ") or line.startswith("\t"):
            curr.append(line.strip())
        else:
            if curr:
                indented_blocks.append("\n".join(curr))
                curr = []
    if curr:
        indented_blocks.append("\n".join(curr))
    return blocks + indented_blocks

def is_sparql(query):
    query_l = query.lower()
    # Look for strong SPARQL clues, *not* SQL/Cypher
    return (
        ("prefix" in query_l or "wdt:" in query_l or "construct" in query_l or "ask" in query_l or "select" in query_l)
        and "{" in query_l
    )

def clean_context(text, code_blocks):
    for code in code_blocks:
        text = text.replace(code, "")
    text = re.sub(r"```(?:sparql)?|```", "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def get_permalink(obj):
    try:
        return "https://reddit.com" + obj.permalink
    except Exception:
        return ""

def safe_author(obj):
    try:
        return str(obj.author) if obj.author else None
    except Exception:
        return None

def safe_getattr(obj, attr, default=None):
    try:
        val = getattr(obj, attr)
        if hasattr(val, "__str__"):
            return str(val)
        return val
    except Exception:
        return default

def process_reddit_item(body, metadata, results, seen_queries):
    code_blocks = extract_code_blocks(body)
    code_blocks = [c for c in code_blocks if is_sparql(c) and len(c) >= MIN_QUERY_LEN]
    if not code_blocks:
        return
    context = clean_context(body, code_blocks)
    for query in code_blocks:
        # --- Exclusion logic: remove only if definitely SQL/Cypher ---
        if is_strongly_not_sparql(query, context):
            continue
        key = (query.strip(), context.strip())
        if key in seen_queries:
            continue
        seen_queries.add(key)
        result = {
            "query": query.strip(),
            "context": context,
        }
        # Make all metadata serializable
        for k, v in metadata.items():
            result[k] = str(v) if v is not None else None
        results.append(result)

all_results = []
seen_queries = set()

for sub in SUBREDDITS:
    print(f"===[ Scraping /r/{sub} ]===")
    subreddit = reddit.subreddit(sub)
    # --- Submissions ---
    for term in SEARCH_TERMS:
        try:
            for submission in tqdm(subreddit.search(term, sort="new", limit=MAX_POSTS_PER_SUB), desc=f"{sub}-search-{term}"):
                body = (submission.selftext or "") + "\n" + (submission.title or "")
                metadata = {
                    "subreddit": sub,
                    "author": safe_author(submission),
                    "permalink": get_permalink(submission),
                    "created_utc": safe_getattr(submission, "created_utc"),
                    "type": "submission",
                    "score": safe_getattr(submission, "score"),
                    "id": submission.id,
                    "title": safe_getattr(submission, "title"),
                }
                process_reddit_item(body, metadata, all_results, seen_queries)
                time.sleep(0.2)
        except Exception as e:
            print(f"Error searching {sub}/{term}: {e}")
            time.sleep(5)

    # --- Comments (whole subreddit stream) ---
    try:
        for comment in tqdm(subreddit.comments(limit=MAX_COMMENTS_PER_SUB), desc=f"{sub}-comments"):
            body = comment.body
            metadata = {
                "subreddit": sub,
                "author": safe_author(comment),
                "permalink": get_permalink(comment),
                "created_utc": safe_getattr(comment, "created_utc"),
                "type": "comment",
                "score": safe_getattr(comment, "score"),
                "id": comment.id,
                "parent_id": safe_getattr(comment, "parent_id"),
                "link_id": safe_getattr(comment, "link_id"),
            }
            process_reddit_item(body, metadata, all_results, seen_queries)
            time.sleep(0.15)
    except Exception as e:
        print(f"Error fetching comments for {sub}: {e}")
        time.sleep(5)

print(f"Total unique query/context pairs: {len(all_results)}")

with open("reddit_sparql_examples.jsonl", "w", encoding="utf-8") as f:
    for ex in all_results:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("Saved to reddit_sparql_examples.jsonl")
