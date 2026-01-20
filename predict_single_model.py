# predict.py
# Robust Single-Model Predictor
# - Auto-detects 5 folds from a parent folder
# - Handles Windows Unicode errors
# - Matches DeBERTa-v3 config

import os
import json
import csv
import argparse
import numpy as np
import torch
import glob
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# --- Windows Environment Fixes ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_test_data(test_path):
    print(f"[Info] Loading test data from {test_path}...")
    # FIXED: Encoding for Windows
    with open(test_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        
    if isinstance(obj, dict) and "sentences" in obj:
        return list(range(len(obj["sentences"]))), obj["sentences"]
    if isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], dict):
            texts = [x["sentence"] for x in obj]
            ids = [x.get("idx", i) for i, x in enumerate(obj)]
            return ids, texts
        else:
            return list(range(len(obj))), obj
    raise ValueError("Unknown test file format")

def get_fold_dirs(parent_dir):
    # Auto-find subdirectories like 'fold1', 'fold2', 'checkpoint-xyz'
    # Priority: explicitly named fold dirs -> standard checkpoints
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Model directory '{parent_dir}' not found.")
        
    # Check for direct folds (fold1, fold2...)
    folds = [d for d in os.listdir(parent_dir) if "fold" in d and os.path.isdir(os.path.join(parent_dir, d))]
    
    # If no "fold" subfolders, maybe the parent dir IS the model (single fold/model)
    if not folds:
        if any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(parent_dir)):
            print(f"[Info] No sub-folds found. Using parent dir as single model.")
            return [parent_dir]
            
    # Sort folds naturally (fold1, fold2, ..., fold5)
    folds.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 999))
    return [os.path.join(parent_dir, f) for f in folds]

def predict(args):
    ids, texts = load_test_data(args.test_path)
    fold_dirs = get_fold_dirs(args.model_dir)
    
    print(f"[Info] Found {len(fold_dirs)} model checkpoints in {args.model_dir}")
    if len(fold_dirs) == 0:
        raise ValueError("No valid model folders found!")

    # Load Tokenizer from the first fold
    try:
        tok = AutoTokenizer.from_pretrained(fold_dirs[0])
    except:
        tok = AutoTokenizer.from_pretrained(fold_dirs[0], use_fast=False)

    # Prepare Dataset
    ds = Dataset.from_dict({"text": texts})
    
    def _tok(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)
    
    ds = ds.map(_tok, batched=True, desc="Tokenizing")
    ds = ds.remove_columns(["text"]).with_format("torch")
    
    # Inference Loop
    all_logits = []
    
    for f_dir in fold_dirs:
        print(f"--- Predicting with {os.path.basename(f_dir)} ---")
        model = AutoModelForSequenceClassification.from_pretrained(f_dir)
        model.eval()
        
        if torch.cuda.is_available():
            model.cuda()
            
        fold_logits = []
        # Batch size for inference
        bs = 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(ds), bs), desc="Inference"):
                batch = ds[i:i+bs]
                inputs = {k: v.cuda() if torch.cuda.is_available() else v 
                          for k, v in batch.items()}
                outputs = model(**inputs)
                fold_logits.append(outputs.logits.cpu().numpy())
                
        all_logits.append(np.concatenate(fold_logits, axis=0))
        
    # Average Logits
    avg_logits = np.mean(all_logits, axis=0)
    probs = torch.softmax(torch.tensor(avg_logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    
    # Save Submission
    print(f"[Info] Saving submission to {args.out_csv}...")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, preds):
            w.writerow([i, int(p)])
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Parent folder containing fold1, fold2, etc.")
    parser.add_argument("--test_path", type=str, default="test_without_labels.json")
    parser.add_argument("--out_csv", type=str, default="submission.csv")
    parser.add_argument("--max_length", type=int, default=128, help="Must match training length")
    
    args = parser.parse_args()
    predict(args)