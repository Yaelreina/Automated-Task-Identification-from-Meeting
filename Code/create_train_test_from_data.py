import pandas as pd
import json
import os

# Path Configurations
# Note: Paths are set relative to the 'code' directory
# Assuming structure:
# root/
#   Data/
#     transcriptions/
#     tasks/

# 1. Path to Transcripts (CSVs)
CSV_DIR = "../Data/transcriptions"

# 2. Path to Tasks (JSONs)
JSON_DIR = "../Data/tasks"

# 3. Output Directory (saving to the main 'data' folder)
OUTPUT_DIR = "../Data"

# Quantity Settings
NUM_FILES = 230   # Total files to process
TRAIN_SIZE = 200  # First 200 used for training, the rest for testing

# Final Output Filenames
OUTPUT_TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.jsonl')
OUTPUT_TEST_FILE = os.path.join(OUTPUT_DIR, 'test.jsonl')

all_examples = []

print(f"--- Starting Processing ---")
print(f"Looking for CSVs in: {CSV_DIR}")
print(f"Looking for JSONs in: {JSON_DIR}")

for i in range(NUM_FILES):
    # Construct paths for specific files
    csv_filename = os.path.join(CSV_DIR, f'Meeting_{i}.csv')
    json_filename = os.path.join(JSON_DIR, f'Meeting_{i}_tasks.json')

    # Check if files exist
    if not os.path.exists(csv_filename):
        print(f"Skipping index {i}: CSV not found at {csv_filename}")
        continue
    if not os.path.exists(json_filename):
        print(f"Skipping index {i}: JSON not found at {json_filename}")
        continue

    try:
        # 1. Read Transcript (CSV)
        df = pd.read_csv(csv_filename)

        conversation_text = ""
        for index, row in df.iterrows():
            # Safely extract speaker and utterance
            speaker = str(row.get('speaker', row.iloc[2])).strip()
            utterance = str(row.get('utterance', row.iloc[0])).strip()
            conversation_text += f"{speaker}: {utterance}\n"

        # 2. Read Tasks (JSON)
        with open(json_filename, 'r', encoding='utf-8') as f:
            tasks_data = json.load(f)

        # Convert to JSON string
        output_json_string = json.dumps(tasks_data, ensure_ascii=False)

        # 3. Create Example Object
        example = {
            "instruction": "Extract action items from the following meeting transcript into JSON format containing Assignee, Action, and Deadline.",
            "input": conversation_text,
            "output": output_json_string
        }

        all_examples.append(example)
        print(f"Processed Meeting_{i} successfully.")

    except Exception as e:
        print(f"Error processing index {i}: {e}")

# Save to Disk
if all_examples:
    train_data = all_examples[:TRAIN_SIZE]
    test_data = all_examples[TRAIN_SIZE:]

    def save_to_jsonl(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved {len(data)} examples to {filename}")

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_to_jsonl(train_data, OUTPUT_TRAIN_FILE)
    save_to_jsonl(test_data, OUTPUT_TEST_FILE)

    print("\nDone! Output files are in:", OUTPUT_DIR)
else:
    print("\nNo files processed. Please check your directory paths.")