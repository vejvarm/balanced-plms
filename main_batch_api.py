import argparse
import os
import subprocess
import sys

COMMANDS = {
    "prepare_dataset": "scraping/batch_api/00_prepare_dataset.py",
    "prepare_stackexchange": "scraping/batch_api/00_prepare_stackexchange.py",
    "prepare_batch": "scraping/batch_api/01_prepare_batch.py",
    "prepare_batches_stackexchange": "scraping/batch_api/01_prepare_batches_stackexchange.py",
    "submit_batch": "scraping/batch_api/02_submit_batch.py",
    "submit_batches": "scraping/batch_api/02_submit_batches.py",
    "download_results": "scraping/batch_api/03_download_batch_results.py",
    "merge_batch_results": "scraping/batch_api/04_merge_batch_results.py",
    "merge_batch_results_stackexchange": "scraping/batch_api/04_merge_batch_results_stackexchange.py",
    "prepare_arrow_dataset": "scraping/batch_api/05_prepare_arrow_dataset.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run batch API scripts")
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
