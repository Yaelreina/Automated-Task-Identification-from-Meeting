from unsloth import FastLanguageModel
import os
import torch
import pandas as pd
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import unsloth_zoo.fused_losses.cross_entropy_loss

def safe_get_chunk_multiplier(vocab_size, target_gb):
    if target_gb is None or target_gb <= 0.1:
        target_gb = 4.0
    return (vocab_size * 4 / 1024 / 1024 / 1024) / target_gb

unsloth_zoo.fused_losses.cross_entropy_loss._get_chunk_multiplier = safe_get_chunk_multiplier
print(" Patch Applied: ZeroDivisionError fix active.")

# Configuration
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
# Updated to relative path for submission compatibility
DATA_DIR = "../Data" 
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
OUTPUT_DIR = "../Code/outputs"

# 1. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 512,
    load_in_4bit = True,
    dtype = None,
    device_map = {"": 0},
)

# 2. Add LoRA Adapters
# We use Low-Rank Adaptation to fine-tune efficiently
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = True,
)
# 

# 3. Prompt Preparation
# We added {} placeholders for the Instruction, Input, and Response
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN so the model learns when to stop generation!

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Format the text and append the EOS token at the end
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Training Arguments
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = OUTPUT_DIR,
        dataloader_num_workers = 0,
    ),
)
# 

# 5. Start Training
print("--- Starting Training (CORRECTED) ---")
trainer.train()

# 6. Save Model and Logs
history = trainer.state.log_history
df = pd.DataFrame([entry for entry in history if 'loss' in entry])
df.to_csv("loss_history.csv", index=False)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"--- Training Complete! Model saved to {OUTPUT_DIR} ---")