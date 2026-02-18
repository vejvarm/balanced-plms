import openai
import pathlib

openai.api_key = open("../batch_api/.api_key", "r").read().strip()

BATCHES_DIR = pathlib.Path("v2.1_expand/.batches")
file_id_log_path = ".current_file_id.txt"
batch_id_log_path = ".current_batch_queue.txt"

# Load already uploaded file names (for duplicate check)
submitted_files = set()
if pathlib.Path(file_id_log_path).exists():
    with open(file_id_log_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:  # We log: "filename fileid"
                submitted_files.add(parts[0])
            elif len(parts) == 1:  # Legacy: just fileid, no filename
                pass

with open(file_id_log_path, "a") as file_id_log, open(batch_id_log_path, "a") as batch_id_log:
    for batch_file in sorted(BATCHES_DIR.glob("openai_batch_input*.jsonl")):
        fname = str(batch_file.resolve())
        if fname in submitted_files:
            print(f"Skipping {batch_file}, already submitted.")
            continue
        print(f"Uploading {batch_file} ...")
        upload = openai.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        file_id = upload.id
        print("Uploaded file ID:", file_id)
        # Log both filename and file_id for better tracking
        file_id_log.write(f"{fname} {file_id}\n")
        file_id_log.flush()

        batch = openai.batches.create(
            input_file_id=file_id,
            endpoint='/v1/chat/completions',
            completion_window="24h"
        )
        print("Batch ID:", batch.id)
        batch_id_log.write(batch.id + "\n")
        batch_id_log.flush()

print("All new batches submitted! Check progress at https://platform.openai.com/batches")
