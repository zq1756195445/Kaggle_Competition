# train.py
# Grandmaster Edition: DeBERTa-v3 + FGM + LLRD + Label Smoothing
# Includes fixes for: Windows Silent Crash, Trainer Signature, Column Names

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from tqdm import tqdm

# --- CRITICAL WINDOWS & ENVIRONMENT FIXES ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"          # Prevents silent DLL crashes
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppresses symlink warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"       # Prevents deadlock in tokenizers

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
    EarlyStoppingCallback
)

# ------------------------- 1. FGM (Adversarial Training Helper) -------------------------
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # Search for embedding parameters and add noise
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # Restore original weights
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ------------------------- 2. Custom Trainer (Adversarial) -------------------------
class AdversarialTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgm = FGM(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to handle label smoothing correctly without modifying inputs.
        """
        labels = inputs.get("labels")
        # Create inputs without labels
        inputs_without_labels = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs_without_labels)
        if labels is not None:
            logits = outputs["logits"]
            reduction = 'sum' if num_items_in_batch is not None else 'mean'
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor, reduction=reduction)
            loss = loss_fct(logits, labels)
            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch
        else:
            # If no labels (unlikely in training), fall back
            loss = outputs["loss"] if "loss" in outputs else None
        return (loss, outputs) if return_outputs else loss

    # UPDATED SIGNATURE: Includes 'num_items_in_batch' to fix TypeError on newer Transformers
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 1. Normal Forward & Backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
            
        self.accelerator.backward(loss)
        
        # 2. FGM Attack (Adversarial Step)
        self.fgm.attack()
        
        with self.compute_loss_context_manager():
            loss_adv = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            
        if self.args.n_gpu > 1:
            loss_adv = loss_adv.mean()
            
        # Accumulate adversarial gradients
        self.accelerator.backward(loss_adv)
        
        # 3. Restore original embeddings
        self.fgm.restore()
        
        return loss.detach()

# ------------------------- 3. LLRD (Layer-wise Learning Rate) -------------------------
def get_llrd_optimizer_params(model, base_lr, weight_decay, decay_factor=0.9):
    """
    Assigns lower LR to bottom layers and higher LR to top layers.
    Logic tailored for DeBERTa-v3 architecture.
    """
    named_parameters = list(model.named_parameters())
    optimizer_grouped_parameters = []
    
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        
        lr = base_lr
        
        # DeBERTa structure: embeddings -> encoder.layer.0 ... -> classifier
        if "embeddings" in name:
            lr = base_lr * (decay_factor ** 13)
        elif "encoder.layer" in name:
            try:
                # Example name: "deberta.encoder.layer.11.output..."
                parts = name.split("encoder.layer.")
                if len(parts) > 1:
                    layer_num = int(parts[1].split(".")[0])
                    # Layer 11 (top) is closer to output, so higher LR
                    layers_from_top = 11 - layer_num
                    lr = base_lr * (decay_factor ** (layers_from_top + 1))
            except:
                lr = base_lr
        elif "classifier" in name or "pooler" in name:
            lr = base_lr
            
        optimizer_grouped_parameters.append({
            "params": [param],
            "lr": lr,
            "weight_decay": weight_decay
        })

    return optimizer_grouped_parameters

# ------------------------- 4. Config & Utils -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="train.json")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, default="./output_model")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16) 
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_and_flatten(train_path):
    # Added encoding='utf-8' to fix Windows Unicode errors
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels, ids = [], [], []
    for item in data:
        texts.append(item["sentence"])
        labels.append(int(item["label"]))
        ids.append(item.get("idx", len(ids)))
    return texts, labels, ids

def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    y_true = eval_pred.label_ids
    return {"acc": accuracy_score(y_true, preds), "f1": f1_score(y_true, preds)}

# ------------------------- 5. Main Execution -------------------------
def main():
    print("[Info] Script starting...", flush=True)
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # 1. Load Data
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"{args.train_path} not found!")
    
    texts, labels, ids = load_and_flatten(args.train_path)
    print(f"[Info] Loaded {len(texts)} samples.", flush=True)
    
    # 2. Tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        print("[Warning] Fast tokenizer failed, falling back to slow version.")
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        
    # 3. Build Dataset
    ds = Dataset.from_dict({"text": texts, "labels": labels, "idx": ids})
    
    def _tok(batch):
        return tok(batch["text"], truncation=True, padding=False, max_length=args.max_length)
    
    ds = ds.map(_tok, batched=True, desc="Tokenizing")
    ds = ds.remove_columns(["text"])

    # --- FIX: Rename 'label' to 'labels' ---
    # This prevents the "ValueError: The model did not return a loss" error
    if "label" in ds.column_names:
        print("[Info] Renaming 'label' to 'labels'...", flush=True)
        ds = ds.rename_column("label", "labels")
    # ---------------------------------------

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_preds = np.zeros(len(labels))
    fold_metrics = []

    print("\n[Info] Starting Training Loop (FGM + LLRD enabled)...")
    fold_iterator = tqdm(skf.split(range(len(labels)), labels), total=args.folds, desc="Overall Progress")

    for fold, (tr_idx, va_idx) in enumerate(fold_iterator, start=1):
        tqdm.write(f"\n--- Fold {fold}/{args.folds} ---")
        fold_dir = os.path.join(args.output_dir, f"fold{fold}")
        
        train_ds = ds.select(tr_idx)
        valid_ds = ds.select(va_idx)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        
        # --- LLRD Optimizer ---
        # We manually create the optimizer to apply specific LRs to specific layers
        opt_params = get_llrd_optimizer_params(model, base_lr=2e-5, weight_decay=0.01, decay_factor=0.9)
        optimizer = torch.optim.AdamW(opt_params)
        
        targs = TrainingArguments(
            output_dir=fold_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size*2,
            num_train_epochs=args.epochs,
            fp16=torch.cuda.is_available(),
            label_names=["labels"],
            
            # Strategy
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            save_total_limit=1,
            
            # Windows Stability
            dataloader_num_workers=0, 
            report_to="none",
            
            # Regularization
            label_smoothing_factor=0.1,
            warmup_ratio=0.1,
        )

        # Check for existing checkpoints to resume
        resume_from_checkpoint = None
        if os.path.exists(fold_dir):
            checkpoints = [d for d in os.listdir(fold_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                resume_from_checkpoint = os.path.join(fold_dir, checkpoints[-1])
                print(f"[Info] Resuming from checkpoint: {resume_from_checkpoint}")

        # Use Custom AdversarialTrainer
        trainer = AdversarialTrainer(
            model=model, 
            args=targs, 
            train_dataset=train_ds, 
            eval_dataset=valid_ds,
            processing_class=tok, 
            data_collator=data_collator, 
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None), # Pass custom optimizer, let Trainer make scheduler
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save best model and predict
        trainer.save_model(fold_dir)
        
        val_preds = trainer.predict(valid_ds)
        # Class 1 probability
        val_probs = torch.softmax(torch.tensor(val_preds.predictions), dim=-1)[:, 1].numpy()
        
        # Map back to original indices
        val_indices = valid_ds["idx"]
        oof_preds[val_indices] = val_probs
        
        fold_score = val_preds.metrics["test_f1"]
        fold_metrics.append(fold_score)
        tqdm.write(f"Fold {fold} F1: {fold_score:.4f}")

    # 5. Save Results
    mean_f1 = np.mean(fold_metrics)
    print(f"\n\n===== Training Complete =====")
    print(f"Mean CV F1: {mean_f1:.4f}")

    df_oof = pd.DataFrame({"id": ids, "label": labels, "pred_prob": oof_preds})
    df_oof = df_oof.sort_values("id").reset_index(drop=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    df_oof.to_csv(os.path.join(args.output_dir, "oof_predictions.csv"), index=False)
    print(f"[Done] Saved OOF predictions.")

if __name__ == "__main__":
    main()