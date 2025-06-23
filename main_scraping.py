import argparse
import os
import subprocess
import sys

COMMANDS = {
    "c4": "scraping/scrape_c4.py",
    "reddit": "scraping/scrape_reddit.py",
    "stackoverflow": "scraping/scrape_stackoverflow.py",
    "wikibooks": "scraping/scrape_wikibooks.py",
    "wikibase_rdf": "scraping/scrape_wikibase_rdf.py",
    "wikidata_examples": "scraping/scrape_wikidata_examples.py",
    "wikidata_tutorial": "scraping/scrape_wikidata_sparql_tutorial.py",
    "wikidata_queries": "scraping/scrape_wikidata_SPARQL_query_service_queries_examples.py",
    "fine_tune_explainer": "scraping/fine_tune_explainer.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run scraping scripts")
    parser.add_argument("task", nargs="?", help="Task to run")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for the script",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_tasks", help="List available tasks"
    )
    args = parser.parse_args()

    if args.list_tasks or args.task is None:
        print("Available tasks:")
        for name in COMMANDS:
            print("  " + name)
        if args.task is None:
            return
        sys.exit(0)

    script = COMMANDS.get(args.task)
    if script is None:
        print(f"Unknown task: {args.task}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run([sys.executable, script, *args.extra_args], check=True, env=env)


if __name__ == "__main__":
    main()
