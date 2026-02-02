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

# Definitions
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
DATA_DIR = "../Data"
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
REPORT_FILE = "baseline_detailed_report.txt"
CSV_FILE = "baseline_analysis_results.csv"

# Loading Models
print("--- Loading Base Model (Llama-3-8B) ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    device_map={"": 0}
)
FastLanguageModel.for_inference(model)

print("--- Loading AlephBERT for Semantic Matching ---")
embedder = SentenceTransformer('imvladikon/sentence-transformers-alephbert')


# Helper Functions (Metrics - identical to the final model)

def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
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

        # Assignee (Fuzzy)
        a_pred = str(pred.get('assignee', ''))
        a_true = str(true.get('assignee', ''))
        is_same_person = fuzz.ratio(a_pred, a_true) >= 80 or a_pred in a_true or a_true in a_pred
        assignee_scores.append(1.0 if is_same_person else 0.0)

        # Action (Semantic)
        action_scores.append(max(0.0, similarity_matrix[i, j]))

        # Deadline (Soft Match)
        d_pred = normalize_text(pred.get('deadline', ''))
        d_true = normalize_text(true.get('deadline', ''))
        match = (d_pred in d_true) or (d_true in d_pred) or (fuzz.ratio(d_pred, d_true) > 85)
        if (not d_pred and d_true) or (d_pred and not d_true): match = False
        deadline_scores.append(1.0 if match else 0.0)

    denominator = max(len(true_items), len(pred_items))

    return (np.sum(assignee_scores) / denominator,
            np.sum(action_scores) / denominator,
            np.sum(deadline_scores) / denominator)


# Robust Extraction

def extract_robust_json(text):
    """
    Tries to extract JSON at all costs:
    1. Cleans Markdown
    2. Searches for {} or [] boundaries
    3. Tries json.loads
    4. Tries ast.literal_eval (in case of single quotes)
    5. Tries raw_decode (in case of trailing garbage)
    """
    text = text.replace("```json", "").replace("```", "").strip()

    # 1. Search start
    start_brace = text.find('{')
    start_bracket = text.find('[')

    if start_brace == -1 and start_bracket == -1: return None

    start = start_brace if (
            start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket)) else start_bracket

    # 2. Search end (the last matching bracket)
    try:
        end = max(text.rfind('}'), text.rfind(']')) + 1
        candidate = text[start:end]

        # Strategy 1: Standard JSON
        try:
            return json.loads(candidate)
        except:
            pass

        # Strategy 2: Python format (ast) - critical for base models!
        try:
            return ast.literal_eval(candidate)
        except:
            pass

        # Strategy 3: raw_decode (cleaning trailing garbage)
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(candidate)
            return obj
        except:
            pass

        return None
    except:
        return None


def force_translate_to_hebrew(json_obj):
    """Translates only English values to Hebrew"""
    if isinstance(json_obj, list):
        return [force_translate_to_hebrew(item) for item in json_obj]
    if isinstance(json_obj, dict):
        new_obj = {}
        translator = GoogleTranslator(source='auto', target='iw')
        for k, v in json_obj.items():
            if isinstance(v, str) and k in ['action', 'assignee', 'deadline']:
                try:
                    # Translate only if there are English letters
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


# Main Execution

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Extract action items from the conversation into a JSON format. Use specific keys: 'assignee', 'action', 'deadline'.

### Input:
{}

### Response:
"""

print("--- Starting Baseline Evaluation (Robust Mode) ---")
dataset = load_dataset("json", data_files={"test": TEST_FILE}, split="test")

metrics_agg = {"assignee": [], "action": [], "deadline": []}
detailed_results = []
failures = 0

for i, row in tqdm(enumerate(dataset), total=len(dataset)):
    prompt = alpaca_prompt.format(row['input'])

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    decoded = tokenizer.batch_decode(outputs)[0]

    # Extraction
    raw_response = decoded.split("### Response:")[-1]

    # Using the new robust extraction
    parsed_data = extract_robust_json(raw_response)

    if parsed_data is None:
        metrics_agg["assignee"].append(0)
        metrics_agg["action"].append(0)
        metrics_agg["deadline"].append(0)
        failures += 1
        continue

    try:
        # Normalize to uniform structure
        if isinstance(parsed_data, list):
            items = parsed_data
        elif isinstance(parsed_data, dict):
            items = parsed_data.get("action_items", [])
            # Sometimes it's a dictionary inside a dictionary by mistake
            if not items and parsed_data:
                for v in parsed_data.values():
                    if isinstance(v, list):
                        items = v
                        break
        else:
            items = []

        # Translation (because the baseline answers in English)
        translated_items = force_translate_to_hebrew(items)

        true_json = json.loads(row['output'])

        # Calculating metrics
        s_assignee, s_action, s_deadline = calculate_complex_metrics(
            translated_items,
            true_json.get('action_items', [])
        )

        metrics_agg["assignee"].append(s_assignee)
        metrics_agg["action"].append(s_action)
        metrics_agg["deadline"].append(s_deadline)

        # Save to CSV
        detailed_results.append({
            "Sample ID": i + 1,
            "Assignee Score": round(s_assignee, 2),
            "Action Score": round(s_action, 2),
            "Deadline Score": round(s_deadline, 2),
            "Prediction (Translated)": json.dumps(translated_items, ensure_ascii=False),
            "Ground Truth": json.dumps(true_json.get('action_items'), ensure_ascii=False)
        })

    except Exception as e:
        print(f"Error sample {i}: {e}")
        metrics_agg["assignee"].append(0)
        metrics_agg["action"].append(0)
        metrics_agg["deadline"].append(0)
        failures += 1

# Summary and Save
print(f"\nTotal Failures: {failures}/{len(dataset)}")

final_summary = f"""
BASELINE MULTI-METRIC REPORT
===============================
(Evaluated using: Hungarian Matching + AlephBERT + Translation + Robust Parsing)

1. Assignee Accuracy: {np.mean(metrics_agg['assignee']) * 100:.1f}%
2. Action Similarity: {np.mean(metrics_agg['action']) * 100:.1f}%
3. Deadline Accuracy: {np.mean(metrics_agg['deadline']) * 100:.1f}%
"""

print(final_summary)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(final_summary)

if detailed_results:
    keys = detailed_results[0].keys()
    with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(detailed_results)
    print(f"Baseline Excel saved to: {CSV_FILE}")