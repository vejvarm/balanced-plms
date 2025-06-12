import openai

openai.api_key = open(".api_key", "r").read()

# 1. Upload the file
upload = openai.files.create(
    file=open(".openai_batch_input.jsonl", "rb"),
    purpose="batch"
)
file_id = upload.id
print("Uploaded file ID:", file_id)
with open(".current_file_id.txt", "a") as f:
    f.write(file_id+"\n")

# 2. Create the batch
batch = openai.batches.create(
    input_file_id=file_id,
    endpoint='/v1/chat/completions',
    completion_window="24h"
)
print("Batch ID:", batch.id)
with open(".current_batch_queue.txt", "a") as f:
    f.write(batch.id+"\n")
# check progress at https://platform.openai.com/batches

