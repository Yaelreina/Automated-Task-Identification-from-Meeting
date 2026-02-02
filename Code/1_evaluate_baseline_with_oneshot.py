from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import re
import csv
import ast
import numpy as np
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
DATA_DIR = "../Data" 
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
REPORT_FILE = "baseline_oneshot_report.txt"
CSV_FILE = "baseline_oneshot_results.csv"

# Load Models
print("--- Loading Base Model (Llama-3-8B) ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    device_map={"": 0}
)
FastLanguageModel.for_inference(model)

print("--- Loading AlephBERT ---")
embedder = SentenceTransformer('imvladikon/sentence-transformers-alephbert')


# Metrics
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    # Removing common Hebrew prepositions and the word "day"
    text = re.sub(r'\b(ב|ל|ה|את|של|ביום|יום)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def calculate_complex_metrics(pred_items, true_items):
    if not pred_items and not true_items: return 1.0, 1.0, 1.0
    if not pred_items or not true_items: return 0.0, 0.0, 0.0

    pred_actions = [p.get('action', '') for p in pred_items]
    true_actions = [t.get('action', '') for t in true_items]

    pred_vecs = embedder.encode(pred_actions)
    true_vecs = embedder.encode(true_actions)

    if len(pred_vecs) == 0 or len(true_vecs) == 0: return 0.0, 0.0, 0.0

    similarity_matrix = cosine_similarity(pred_vecs, true_vecs)
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    assignee_scores = []
    action_scores = []
    deadline_scores = []

    for i, j in zip(row_ind, col_ind):
        pred = pred_items[i]
        true = true_items[j]
        # Assignee Metric
        a_pred = str(pred.get('assignee', ''))
        a_true = str(true.get('assignee', ''))
        is_same_person = fuzz.ratio(a_pred, a_true) >= 80 or a_pred in a_true or a_true in a_pred
        assignee_scores.append(1.0 if is_same_person else 0.0)
        # Action Metric
        action_scores.append(max(0.0, similarity_matrix[i, j]))
        # Deadline Metric
        d_pred = normalize_text(pred.get('deadline', ''))
        d_true = normalize_text(true.get('deadline', ''))
        match = (d_pred in d_true) or (d_true in d_pred) or (fuzz.ratio(d_pred, d_true) > 85)
        if (not d_pred and d_true) or (d_pred and not d_true): match = False
        deadline_scores.append(1.0 if match else 0.0)

    denominator = max(len(true_items), len(pred_items))
    return (np.sum(assignee_scores) / denominator,
            np.sum(action_scores) / denominator,
            np.sum(deadline_scores) / denominator)


# Extraction and Translation
def extract_robust_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    start_brace = text.find('{')
    start_bracket = text.find('[')
    if start_brace == -1 and start_bracket == -1: return None
    start = start_brace if (
                start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket)) else start_bracket
    try:
        end = max(text.rfind('}'), text.rfind(']')) + 1
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except:
            try:
                return ast.literal_eval(candidate)
            except:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(candidate)
                return obj
    except:
        return None


def force_translate_to_hebrew(json_obj):
    if isinstance(json_obj, list):
        return [force_translate_to_hebrew(item) for item in json_obj]
    if isinstance(json_obj, dict):
        new_obj = {}
        translator = GoogleTranslator(source='auto', target='iw')
        for k, v in json_obj.items():
            if isinstance(v, str) and k in ['action', 'assignee', 'deadline']:
                try:
                    if re.search(r'[a-zA-Z]', v):
                        new_obj[k] = translator.translate(v)
                    else:
                        new_obj[k] = v
                except:
                    new_obj[k] = v
            else:
                new_obj[k] = v
        return new_obj
    return json_obj


# One-Shot Prompt
alpaca_oneshot_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Extract action items from the conversation into a JSON format. Use specific keys: 'assignee', 'action', 'deadline'.

### Input:
דני: יוסי, תכין את המצגת למחר.
יוסי: אין בעיה.

### Response:
[{{ "assignee": "Yossi", "action": "Prepare presentation", "deadline": "Tomorrow" }}]

### Instruction:
Extract action items from the conversation into a JSON format. Use specific keys: 'assignee', 'action', 'deadline'.

### Input:
{}

### Response:
"""

print("--- Starting Baseline One-Shot Evaluation ---")
dataset = load_dataset("json", data_files={"test": TEST_FILE}, split="test")

metrics_agg = {"assignee": [], "action": [], "deadline": []}
detailed_results = []

for i, row in tqdm(enumerate(dataset), total=len(dataset)):
    # Formatting will work because of the double braces
    prompt = alpaca_oneshot_prompt.format(row['input'])

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    decoded = tokenizer.batch_decode(outputs)[0]

    raw_response = decoded.split("### Response:")[-1]
    parsed_data = extract_robust_json(raw_response)

    if parsed_data is None:
        metrics_agg["assignee"].append(0)
        metrics_agg["action"].append(0)
        metrics_agg["deadline"].append(0)
        continue

    try:
        if isinstance(parsed_data, list):
            items = parsed_data
        elif isinstance(parsed_data, dict):
            items = parsed_data.get("action_items", [])
        else:
            items = []

        # Mandatory Translation - The baseline outputs in English
        translated_items = force_translate_to_hebrew(items)
        true_json = json.loads(row['output'])

        s_assignee, s_action, s_deadline = calculate_complex_metrics(translated_items,
                                                                     true_json.get('action_items', []))

        metrics_agg["assignee"].append(s_assignee)
        metrics_agg["action"].append(s_action)
        metrics_agg["deadline"].append(s_deadline)

        detailed_results.append({
            "Sample ID": i + 1,
            "Assignee Score": round(s_assignee, 2),
            "Action Score": round(s_action, 2),
            "Deadline Score": round(s_deadline, 2),
            "Prediction": json.dumps(translated_items, ensure_ascii=False),
            "Ground Truth": json.dumps(true_json.get('action_items'), ensure_ascii=False)
        })
    except:
        metrics_agg["assignee"].append(0)
        metrics_agg["action"].append(0)
        metrics_agg["deadline"].append(0)

# Summary
final_summary = f"""
BASELINE (ONE-SHOT) REPORT
=============================
1. Assignee: {np.mean(metrics_agg['assignee']) * 100:.1f}%
2. Action:   {np.mean(metrics_agg['action']) * 100:.1f}%
3. Deadline: {np.mean(metrics_agg['deadline']) * 100:.1f}%
"""
print(final_summary)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(final_summary)

if detailed_results:
    keys = detailed_results[0].keys()
    with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(detailed_results)
    print(f"Baseline Excel saved to: {CSV_FILE}")