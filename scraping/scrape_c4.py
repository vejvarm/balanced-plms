import json
import re
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm
from itertools import islice
import os
from datetime import datetime

def log(msg: str):
    with open("c4_progress.log", "a") as f:
        f.write(f"{datetime.now()}: {msg}\n")

# --- Estimated document counts for C4 variants ---
C4_LENGTHS = {
    "en": 365_000_000,
    "en.noblocklist": 395_000_000,
    "en.noclean": 1_100_000_000
}

FORM_KEYWORDS_RE = re.compile(r"\b(select|ask|construct|describe)\b")
PREFIX_RE = re.compile(r"prefix\s+[^\s:]+:\s*<[^>]+>", re.I)
TRIPLE_PATTERN_RE = re.compile(r"\?\w+\s+\S+\s+\?\w+\s*\.")

def looks_like_sparql(sample):
    text = sample["text"].lower()
    if not (FORM_KEYWORDS_RE.search(text) or PREFIX_RE.search(text)):
        return False
    if "{" not in text or "}" not in text:
        return False
    contents = re.findall(r"\{(.*?)\}", text, re.DOTALL)
    if not any(TRIPLE_PATTERN_RE.search(c) for c in contents):
        return False
    return True

def main(
    outfile="c4_sparql.jsonl",
    max_examples=None,
    variant="en.noclean",
    checkpoint_file="c4_progress.csv",
    streaming=True,
    num_proc=None,
    resume_download=False,
):
    """Scrape SPARQL-like documents from the C4 dataset.

    When ``streaming`` is ``True`` (default), the dataset is streamed and the
    script resumes progress via ``checkpoint_file``. When ``streaming`` is
    ``False``, the dataset is loaded entirely and filtered in parallel using
    ``datasets.Dataset.filter`` before being written out.
    """

    print(f"Loading C4 dataset with variant: {variant}, streaming={streaming} ...")
    ds = load_dataset("allenai/c4", variant, split="train", streaming=streaming, download_config=DownloadConfig(resume_download=resume_download, num_proc=num_proc))
    total = C4_LENGTHS.get(variant, None)
    if total:
        print(f"Estimated documents in split: {total}")
    else:
        print("Estimated documents in split: Unknown")

    if streaming:
        # --- Load last progress (document index, saved count) ---
        finished = False
        while not finished:
            idx = 0
            saved = 0
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file) as f:
                        vals = f.read().strip().split(",")
                        idx = int(vals[0]) - 1
                        saved = int(vals[1]) if len(vals) > 1 else 0
                    print(f"Resuming from doc {idx}, previously saved {saved}")
                except Exception as e:
                    print(f"Could not read checkpoint file, starting from scratch: {e}")

            # --- Iterate from last doc using islice ---
            pbar = tqdm(total=total, initial=idx, unit="docs", desc="Scanning C4", dynamic_ncols=True)
            try:
                for idx, sample in enumerate(islice(ds, idx, None), start=idx):
                    pbar.update(1)
                    if looks_like_sparql(sample):
                        with open(outfile, "a", encoding="utf-8") as fout:
                            fout.write(f"{json.dumps(sample, ensure_ascii=False)}\n")
                        saved += 1
                        pbar.set_postfix(saved=saved)
                    # Save checkpoint every 500000 docs or on match
                    if idx % 500000 == 0 or (max_examples and saved >= max_examples):
                        with open(checkpoint_file, "w") as fcp:
                            fcp.write(f"{idx+1},{saved}")
                    if max_examples and saved >= max_examples:
                        print("Reached max_examples cap, exiting.")
                        break
            except KeyboardInterrupt:
                msg = f"({idx+1},{saved}) Interrupted, saving progress..."
                log(msg)
                print(msg)
                finished = True
            except StopIteration:
                msg = f"({idx+1},{saved}) Finished, saving progress..."
                log(msg)
                print(msg)
                finished = True
            except Exception as e:
                msg = f"({idx+1},{saved}) General Exception {repr(e)}"
                log(msg)
                print(msg)
            finally:
                with open(checkpoint_file, "w") as fcp:
                    fcp.write(f"{idx+1},{saved}")
                print(f"Checkpoint saved at doc {idx+1}, saved={saved}")

        print(f"Finished! Total saved: {saved}")

    else:
        num_proc = num_proc or os.cpu_count()
        print(f"Filtering dataset with num_proc={num_proc} ...")
        filtered_ds = ds.filter(looks_like_sparql, num_proc=num_proc)
        if max_examples:
            filtered_ds = filtered_ds.select(range(min(max_examples, len(filtered_ds))))
        saved = len(filtered_ds)
        print(f"Saving {saved} documents to {outfile} ...")
        filtered_ds.to_json(outfile, num_proc=num_proc)
        print(f"Finished! Total saved: {saved}")

if __name__ == "__main__":
    variant = "en"
    streaming = False
    num_proc = os.cpu_count()
    main(variant=variant, streaming=streaming, num_proc=num_proc)
