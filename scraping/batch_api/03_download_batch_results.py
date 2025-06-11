import time
import openai

batch_id = open(".current_batch_id.txt", "r").read()

client = openai.OpenAI(api_key=open(".api_key", "r").read())


while True:
    batch = client.batches.retrieve(batch_id)
    print(f"Status: {batch.status}")
    if batch.status == "completed" and batch.output_file_id:
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id)
        file_content = file_response.text
        with open(".openai_batch_output.jsonl", "wb") as f:
            f.write(file_content)
        print("Downloaded batch output to openai_batch_output.jsonl")
        break
    elif batch.status in ("failed", "cancelled"):
        print(f"Batch ended with status: {batch.status}")
        break
    else:
        time.sleep(60)  # Wait a minute before checking again
