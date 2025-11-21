#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PADBen Task1 - Optimized Logistic Regression Baseline (PyCharm-ready)

Key ideas:
1) Stronger TF-IDF: word n-grams + character n-grams
2) Light feature engineering: statistical features
3) Proper hyper-parameter search for LR (C, penalty, l1_ratio)
4) Train on full data with best params, then predict test set

How to run:
- Put this file in the same folder as train.json and test_without_labels.json
- Just click Run in PyCharm (no CLI args required)
- Output: submission_lr_optimized.csv
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# =============================================================================
# Config (edit here if your paths differ)
# =============================================================================
TRAIN_FILE = "train.json"
TEST_FILE = "test_without_labels.json"
OUTPUT_FILE = "submission_lr_optimized.csv"

RANDOM_SEED = 42
VAL_SIZE = 0.2

# TF-IDF settings
WORD_MAX_FEATURES = 60000
CHAR_MAX_FEATURES = 40000
WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)

# CV settings
CV_SPLITS = 5


# =============================================================================
# Text cleaning
# =============================================================================
def clean_text_regex(text: str) -> str:
    """
    Simple, safe text cleaning.
    Keep it light to avoid deleting useful signals.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.replace('\\"', '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;!?:'])", r"\1", text)   # remove space before punct
    text = re.sub(r"([.,;!?:'])\s+", r"\1 ", text)  # normalize space after punct
    return text


# =============================================================================
# Statistical features
# =============================================================================
def extract_statistical_features(texts):
    """
    Extracts 8 light statistical features per sentence.
    """
    feats = []
    for t in texts:
        t = t if isinstance(t, str) else str(t)

        char_count = len(t)
        words = t.split()
        word_count = len(words)
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

        punctuation = sum(1 for c in t if c in '.,;!?:\'"-')
        punctuation_ratio = punctuation / char_count if char_count > 0 else 0.0

        upper_count = sum(1 for c in t if c.isupper())
        upper_ratio = upper_count / char_count if char_count > 0 else 0.0

        digit_count = sum(1 for c in t if c.isdigit())
        digit_ratio = digit_count / char_count if char_count > 0 else 0.0

        space_count = sum(1 for c in t if c.isspace())
        space_ratio = space_count / char_count if char_count > 0 else 0.0

        feats.append([
            char_count,
            word_count,
            avg_word_len,
            punctuation,
            punctuation_ratio,
            upper_ratio,
            digit_ratio,
            space_ratio
        ])
    return np.asarray(feats, dtype=np.float32)


# =============================================================================
# Data loading
# =============================================================================
def load_train_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = [clean_text_regex(x["sentence"]) for x in data]
    labels = np.array([x["label"] for x in data], dtype=np.int64)
    ids = [x.get("idx", i) for i, x in enumerate(data)]
    return sentences, labels, ids


def load_test_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = [clean_text_regex(x["sentence"]) for x in data]
    ids = [x["idx"] for x in data]
    return sentences, ids


# =============================================================================
# Feature building
# =============================================================================
def build_features(train_texts, test_texts, use_stats=True):
    """
    Build combined sparse matrix: [word tfidf | char tfidf | stats]
    """
    print("\n[1/3] Building TF-IDF features")

    word_vec = TfidfVectorizer(
        max_features=WORD_MAX_FEATURES,
        ngram_range=WORD_NGRAM,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode"
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        max_features=CHAR_MAX_FEATURES,
        ngram_range=CHAR_NGRAM,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    Xw_train = word_vec.fit_transform(train_texts)
    Xc_train = char_vec.fit_transform(train_texts)

    Xw_test = word_vec.transform(test_texts)
    Xc_test = char_vec.transform(test_texts)

    X_train = hstack([Xw_train, Xc_train])
    X_test = hstack([Xw_test, Xc_test])

    scaler = None
    if use_stats:
        print("[2/3] Adding statistical features")
        Xs_train = extract_statistical_features(train_texts)
        Xs_test = extract_statistical_features(test_texts)

        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(Xs_train)
        Xs_test = scaler.transform(Xs_test)

        X_train = hstack([X_train, Xs_train])
        X_test = hstack([X_test, Xs_test])

    print(f"   Train feature shape: {X_train.shape}")
    print(f"   Test feature shape : {X_test.shape}")
    return X_train, X_test, word_vec, char_vec, scaler


# =============================================================================
# Model training
# =============================================================================
def tune_and_train_lr(X_train, y_train):
    """
    Hyper-parameter tuning for LR using CV.
    We search over penalties that work with 'saga' on sparse data.
    """
    print("\n[3/3] Tuning Logistic Regression with CV")

    base = LogisticRegression(
        solver="saga",
        max_iter=4000,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )

    param_grid = [
        {
            "penalty": ["l2"],
            "C": [0.5, 1.0, 2.0, 4.0],
            "class_weight": [None, "balanced"]
        },
        {
            "penalty": ["l1"],
            "C": [0.5, 1.0, 2.0, 4.0],
            "class_weight": [None, "balanced"]
        },
        {
            "penalty": ["elasticnet"],
            "l1_ratio": [0.15, 0.3, 0.5, 0.7],
            "C": [0.5, 1.0, 2.0, 4.0],
            "class_weight": [None, "balanced"]
        }
    ]

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(
        base,
        param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"   Best CV accuracy: {gs.best_score_:.4f}")
    print(f"   Best params: {gs.best_params_}")

    best = gs.best_estimator_
    return best, gs.best_params_, gs.best_score_


# =============================================================================
# Main
# =============================================================================
def main():
    base_dir = Path(__file__).parent
    train_path = base_dir / TRAIN_FILE
    test_path = base_dir / TEST_FILE

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_FILE} or {TEST_FILE} in {base_dir}."
        )

    print("=" * 70)
    print("PADBen Task1 - Optimized LR Baseline")
    print("=" * 70)

    train_texts, train_labels, _ = load_train_data(train_path)
    test_texts, test_ids = load_test_data(test_path)

    X_train_all, X_test_all, word_vec, char_vec, scaler = build_features(
        train_texts, test_texts, use_stats=True
    )

    # Split for a quick held-out validation report (not used for model selection)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_all, train_labels,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_labels
    )

    best_lr, best_params, best_cv = tune_and_train_lr(X_tr, y_tr)

    # Held-out validation evaluation
    y_va_pred = best_lr.predict(X_va)
    val_acc = accuracy_score(y_va, y_va_pred)
    print("\nHeld-out Validation Accuracy:", f"{val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_va, y_va_pred, digits=4))

    # Retrain on full data
    print("\nRetraining best model on full training set...")
    final_lr = LogisticRegression(
        solver="saga",
        max_iter=4000,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        **{k: v for k, v in best_params.items() if k != "l1_ratio"}  # add l1_ratio separately if present
    )
    if "l1_ratio" in best_params:
        final_lr.set_params(l1_ratio=best_params["l1_ratio"])
    final_lr.fit(X_train_all, train_labels)

    # Predict test
    print("\nPredicting test set...")
    test_pred = final_lr.predict(X_test_all)

    submission = pd.DataFrame({
        "id": test_ids,
        "target": test_pred
    })
    submission.to_csv(base_dir / OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved submission to: {OUTPUT_FILE}")
    print(submission.head(10).to_string(index=False))

    # Small sanity print
    c0 = int((test_pred == 0).sum())
    c1 = int((test_pred == 1).sum())
    print(f"\nPrediction distribution: 0={c0} ({c0/len(test_pred):.2%}), "
          f"1={c1} ({c1/len(test_pred):.2%})")
    print("\nDone.")


if __name__ == "__main__":
    main()
