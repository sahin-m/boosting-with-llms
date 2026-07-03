#!/usr/bin/env python3

import os
import sys
import time
import json
import datetime
import pandas as pd
import openai
from dotenv import load_dotenv

# ===============================
# 🔹 LOGGING HELPERS
# ===============================
def log_batch_id(feature_name, batch_id, log_file="batch_ids.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} | {feature_name} : {batch_id}\n")


def log_status(feature_name, status, elapsed_time, log_dir="../logs"):
    logfile = os.path.join(log_dir, f"{feature_name}.log")

    if os.path.isfile(logfile):
    	with open(logfile, "a") as f:
    	    f.write(f"{datetime.datetime.now()} | Status: {status} | {elapsed_time}\n")
    else:
    	with open(logfile, "w") as f:
    	    f.write(f"{datetime.datetime.now()} | Status: {status} | {elapsed_time}\n")


# ===============================
# 🔹 GPT BATCH FUNCTION
# ===============================
def ask_gpt(batchfile=None, feature_name=None):
    load_dotenv()
    openai.api_key = os.getenv('OPEN_AI_KEY')

    client = openai.OpenAI(api_key=openai.api_key)

    try:
        # Upload batch file
        batch_input_file = client.files.create(
            file=open(batchfile, "rb"),
            purpose="batch"
        )

        # Create batch job
        created_batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window='24h'
        )

        batch_id = created_batch.id
        print(f"Batch created: {batch_id}")

        # Save batch ID
        log_batch_id(feature_name, batch_id)

        current_batch = client.batches.retrieve(batch_id)
        start_time = time.time()

        # Polling loop
        while current_batch.status not in ["completed", "failed", "expired", "cancelled"]:
            elapsed = int(time.time() - start_time)
            print(f"\rStatus: {current_batch.status} | Elapsed: {elapsed}s", end="", flush=True)

            time.sleep(20)
            current_batch = client.batches.retrieve(batch_id)

        print(f"\nFinal Status: {current_batch.status}")

        elapsed_time = f"{int((time.time() - start_time) // 60)} min"

        # Log status
        log_status(feature_name, current_batch.status, elapsed_time)

        # Handle failure cases
        if current_batch.status != "completed":
            print(f" Batch did not complete: {current_batch.status}")
            return None

        # Fetch results
        response = client.files.content(current_batch.output_file_id).text

        output_file = batchfile.replace(".jsonl", "_results.jsonl")

        with open(output_file, "w") as f:
            f.write(response)
            
        print(f"Results saved to: {output_file}")
        return

    except Exception as e:
        print(f"Error: {e}")
        return None


# ===============================
# 🔹 FEATURE PROCESSING
# ===============================
def get_feature_results(feature_name):

    batchfile = f"{feature_name}_batch.jsonl"

    if not os.path.isfile(batchfile):
       print(f"Missing batch file: {batchfile}")
       return

    ask_gpt(
       batchfile=batchfile,
       feature_name=feature_name
    )

    return


# ===============================
# 🔹 ENTRY POINT
# ===============================
if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python llm_feature_runner.py <feature_name>")
        sys.exit(1)

    feature_name = sys.argv[1]

    get_feature_results(feature_name)
