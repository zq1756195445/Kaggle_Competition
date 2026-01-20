# stack_predict.py
import os
import glob
import json
import csv
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Utils for Inference ---
def load_test_data(test_path):
    # Standard loading logic
    with open(test_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Handle list or dict formats
    if isinstance(obj, dict) and "sentences" in obj:
        texts = obj["sentences"]
        ids = list(range(len(texts)))
    elif isinstance(obj, list):
        # Assuming list of dicts with "sentence" or just list of strings
        if len(obj) > 0 and isinstance(obj[0], dict):
            texts = [x["sentence"] for x in obj]
            ids = [x.get("idx", i) for i, x in enumerate(obj)]
        else:
            texts = obj
            ids = list(range(len(texts)))
    return ids, texts

def get_probs_from_model_group(model_parent_dir, texts, batch_size=32):
    """
    Loads all 5 folds from a parent directory (e.g., output_deberta/)
    and averages their probabilities on the test text.
    """
    # 1. Find Checkpoints
    # We look for subfolders like output_deberta/fold1, output_deberta/fold2...
    fold_dirs = [d for d in glob.glob(os.path.join(model_parent_dir, "fold*")) if os.path.isdir(d)]
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {model_parent_dir}")
    
    print(f"Found {len(fold_dirs)} folds in {model_parent_dir}")
    
    # Detect tokenizer from the first fold
    tok = AutoTokenizer.from_pretrained(fold_dirs[0])
    
    # Tokenize Test Set once
    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(lambda x: tok(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    ds = ds.remove_columns(["text"]).with_format("torch")
    
    all_fold_probs = []
    
    for f_dir in fold_dirs:
        print(f"Inferencing: {f_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(f_dir)
        model.eval()
        if torch.cuda.is_available(): model.cuda()
        
        fold_logits = []
        with torch.no_grad():
            for i in range(0, len(ds), batch_size):
                batch = ds[i:i+batch_size]
                inputs = {k: v.cuda() if torch.cuda.is_available() else v 
                          for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
                out = model(**inputs)
                fold_logits.append(out.logits.cpu().numpy())
        
        # Convert logits to Prob(class=1)
        logits_concat = np.concatenate(fold_logits, axis=0)
        probs = torch.softmax(torch.tensor(logits_concat), dim=-1)[:, 1].numpy()
        all_fold_probs.append(probs)
        
    # Average probabilities across 5 folds for this model architecture
    avg_probs = np.mean(all_fold_probs, axis=0)
    return avg_probs

# --- 2. Main Stacking Logic ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_A", type=str, required=True, help="Parent dir of Model A (must contain oof_predictions.csv)")
    ap.add_argument("--dir_B", type=str, required=True, help="Parent dir of Model B (must contain oof_predictions.csv)")
    ap.add_argument("--test_path", type=str, default="test_without_labels.json")
    ap.add_argument("--out_csv", type=str, default="submission.csv")
    args = ap.parse_args()

    print("--- Training Meta-Model (Stacker) ---")
    
    # Load OOFs
    path_a = os.path.join(args.dir_A, "oof_predictions.csv")
    path_b = os.path.join(args.dir_B, "oof_predictions.csv")
    
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    
    # Verification
    if not np.all(df_a["id"] == df_b["id"]):
        raise ValueError("OOF IDs do not match! Did you use the same random seed?")
    if not np.all(df_a["label"] == df_b["label"]):
        raise ValueError("Labels do not match!")
        
    # Prepare Meta-Training Data
    # X = [Prob_A, Prob_B]
    X_train = np.column_stack([df_a["pred_prob"].values, df_b["pred_prob"].values])
    y_train = df_a["label"].values
    
    # Train Logistic Regression
    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_train)
    
    print(f"Stacker Coefficients: Model A: {meta_model.coef_[0][0]:.4f}, Model B: {meta_model.coef_[0][1]:.4f}")
    print(f"Stacker Intercept: {meta_model.intercept_[0]:.4f}")
    
    # Quick check on training score
    meta_preds = meta_model.predict(X_train)
    print(f"Stacker CV F1 Score: {f1_score(y_train, meta_preds):.4f}")
    
    print("\n--- Generating Test Predictions ---")
    ids, texts = load_test_data(args.test_path)
    
    # Get Test Probs for Model A
    print("Getting Model A Test Probs...")
    test_probs_a = get_probs_from_model_group(args.dir_A, texts)
    
    # Get Test Probs for Model B
    print("Getting Model B Test Probs...")
    test_probs_b = get_probs_from_model_group(args.dir_B, texts)
    
    # Stack
    X_test = np.column_stack([test_probs_a, test_probs_b])
    final_preds = meta_model.predict(X_test)
    
    # Save
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, final_preds):
            w.writerow([i, int(p)])
            
    print(f"Done! Saved stacked submission to {args.out_csv}")