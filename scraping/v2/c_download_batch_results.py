import time
import pathlib
import openai

BATCH_RESULT_FOLDER = pathlib.Path("./.batch_results")
BATCH_RESULT_FOLDER.mkdir(exist_ok=True, parents=True)
# batch_ids: list = [b.strip("\n ") for b in open(".current_batch_queue.txt", "r").readlines()]

client = openai.OpenAI(api_key=open(".api_key", "r").read())
batches_not_downloaded = [b.strip(" \n") for b in open(".current_batch_queue.txt", "r").readlines()]
batches_in_progress = set([b.id for b in client.batches.list() if b.status in ("in_progress", "finalizing")] + batches_not_downloaded)
remaining_batches = len(batches_in_progress)


while remaining_batches > 0:
    print(f"Waiting for `{remaining_batches}` batches to complete...")
    for i, batch_id in enumerate(batches_in_progress):
        if batch_id.strip() not in  batches_not_downloaded:
            print(f"Batch `{batch_id}` already downloaded, skipping.")
            continue
        batch = client.batches.retrieve(batch_id)
        print(f"\t Batch `{batch_id}`: {batch.status}")
        if batch.status == "completed" and batch.output_file_id:
            output_file_id = batch.output_file_id
            file_response = client.files.content(output_file_id)
            file_content = file_response.text
            out_file_name = BATCH_RESULT_FOLDER.joinpath(f"openai_batch_output_{batch_id}.jsonl")
            with open(out_file_name, "w") as f:
                f.write(file_content)
            print(f"Downloaded batch output to {out_file_name}")
        elif batch.status in ("failed", "cancelled"):
            print(f"Batch ended with status: {batch.status}")
    batches_in_progress = [b.id for b in client.batches.list() if b.status in ("in_progress", "finalizing")]
    remaining_batches = len(batches_in_progress)
    time.sleep(60) # Wait 60 seconds before checking batches again
