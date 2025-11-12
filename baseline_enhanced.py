#!/usr/bin/env python3
"""
PADBen Task1 - Enhanced Baseline with Feature Engineering
融合文本清理、统计特征、TF-IDF的增强版baseline
"""

import json
import argparse
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 文本清理
# =============================================================================

def clean_text_regex(text):
    """
    文本清理：
    1. 统一引号格式
    2. 规范化空格
    3. 修复标点符号间距
    """
    if not isinstance(text, str):
        return text

    text = text.replace('\"', '"')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'"\s(.*?)\s"', r'"\1"', text)
    text = re.sub(r"\s+([.,;!?:'])", r'\1', text)
    text = re.sub(r'\s*\/\s*', '/', text)

    return text


# =============================================================================
# 特征提取
# =============================================================================

def extract_statistical_features(texts):
    """
    提取统计特征：
    - 句子长度（字符数、单词数）
    - 平均词长
    - 标点符号数量
    - 大写字母比例
    - 数字比例
    """
    features = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text)

        # 基本统计
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        # 标点符号
        punctuation = sum(1 for c in text if c in '.,;!?:\'"-')
        punctuation_ratio = punctuation / char_count if char_count > 0 else 0

        # 大写字母
        upper_count = sum(1 for c in text if c.isupper())
        upper_ratio = upper_count / char_count if char_count > 0 else 0

        # 数字
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / char_count if char_count > 0 else 0

        # 空格比例
        space_count = sum(1 for c in text if c.isspace())
        space_ratio = space_count / char_count if char_count > 0 else 0

        features.append([
            char_count,
            word_count,
            avg_word_len,
            punctuation,
            punctuation_ratio,
            upper_ratio,
            digit_ratio,
            space_ratio
        ])

    return np.array(features)


# =============================================================================
# 数据加载
# =============================================================================

def load_train_data(file_path):
    """加载训练数据并展开sentence_pair格式"""
    print(f"Loading training data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    labels = []

    for item in data:
        sentences.extend(item['sentence_pair'])
        labels.extend(item['label_pair'])

    print(f"✓ Loaded {len(data)} pairs → {len(sentences)} sentences")
    print(f"  Label distribution: 0={labels.count(0)}, 1={labels.count(1)}")

    return sentences, labels


def load_test_data(file_path):
    """加载测试数据（单个句子格式）"""
    print(f"\nLoading test data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = [item['sentence'] for item in data]
    ids = [item['idx'] for item in data]

    print(f"✓ Loaded {len(sentences)} test sentences")
    return sentences, ids


# =============================================================================
# 训练模型
# =============================================================================

def train_enhanced_model(train_sentences, train_labels,
                         max_features=5000, n_estimators=100,
                         use_cleaning=True, use_stats_features=True):
    """
    训练增强版模型：
    1. 可选的文本清理
    2. TF-IDF特征
    3. 统计特征
    4. Random Forest分类
    """
    print("\n" + "=" * 60)
    print("ENHANCED TRAINING PHASE")
    print("=" * 60)

    # Step 1: 文本清理
    if use_cleaning:
        print(f"\n[1/5] Cleaning text...")
        train_sentences = [clean_text_regex(s) for s in train_sentences]
        print(f"✓ Text cleaned")
    else:
        print(f"\n[1/5] Skipping text cleaning")

    # Step 2: TF-IDF特征
    print(f"\n[2/5] Extracting TF-IDF features (max_features={max_features})...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
    print(f"✓ TF-IDF matrix shape: {X_tfidf.shape}")

    # Step 3: 统计特征
    if use_stats_features:
        print(f"\n[3/5] Extracting statistical features...")
        X_stats = extract_statistical_features(train_sentences)

        # 标准化统计特征
        scaler = StandardScaler()
        X_stats = scaler.fit_transform(X_stats)

        print(f"✓ Statistical features shape: {X_stats.shape}")
        print(f"  Features: char_count, word_count, avg_word_len, punctuation,")
        print(f"            punctuation_ratio, upper_ratio, digit_ratio, space_ratio")

        # 合并特征
        print(f"\n[4/5] Combining features...")
        X_combined = hstack([X_tfidf, X_stats])
        print(f"✓ Combined feature matrix shape: {X_combined.shape}")
    else:
        print(f"\n[3/5] Skipping statistical features")
        X_combined = X_tfidf
        scaler = None

    # Step 4: 训练Random Forest
    print(f"\n[5/5] Training Random Forest (n_estimators={n_estimators})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    # 验证集评估
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_combined, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    clf.fit(X_train_split, y_train_split)
    y_pred = clf.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, y_pred)

    print(f"\n{'Validation Accuracy:':<25} {val_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val_split, y_pred, target_names=['Class 0', 'Class 1']))

    # 使用全部数据重新训练
    print("\n[Final] Retraining on full dataset...")
    clf_final = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf_final.fit(X_combined, train_labels)
    print("✓ Final model trained")

    return clf_final, tfidf_vectorizer, scaler, use_cleaning, use_stats_features


# =============================================================================
# 预测
# =============================================================================

def predict_and_save(clf, tfidf_vectorizer, scaler, test_sentences, test_ids,
                     output_file, use_cleaning, use_stats_features):
    """对测试集进行预测并保存"""
    print("\n" + "=" * 60)
    print("PREDICTION PHASE")
    print("=" * 60)

    # Step 1: 文本清理
    if use_cleaning:
        print(f"\n[1/3] Cleaning test text...")
        test_sentences = [clean_text_regex(s) for s in test_sentences]
        print(f"✓ Text cleaned")

    # Step 2: TF-IDF特征
    print(f"\n[2/3] Extracting TF-IDF features...")
    X_test_tfidf = tfidf_vectorizer.transform(test_sentences)
    print(f"✓ TF-IDF matrix shape: {X_test_tfidf.shape}")

    # Step 3: 统计特征
    if use_stats_features:
        print(f"      Extracting statistical features...")
        X_test_stats = extract_statistical_features(test_sentences)
        X_test_stats = scaler.transform(X_test_stats)

        print(f"      Combining features...")
        X_test_combined = hstack([X_test_tfidf, X_test_stats])
        print(f"✓ Combined feature matrix shape: {X_test_combined.shape}")
    else:
        X_test_combined = X_test_tfidf

    # Step 4: 预测
    print(f"\n[3/3] Making predictions...")
    predictions = clf.predict(X_test_combined)
    print(f"✓ Generated {len(predictions)} predictions")
    print(f"  Prediction distribution: 0={list(predictions).count(0)}, 1={list(predictions).count(1)}")

    # 保存
    submission = pd.DataFrame({
        'id': test_ids,
        'target': predictions
    })
    submission.to_csv(output_file, index=False)
    print(f"\n✅ Submission saved to: {output_file}")
    print(f"   Total predictions: {len(submission)}")
    print("\nFirst 10 predictions:")
    print(submission.head(10).to_string(index=False))


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PADBen Task1 - Enhanced Baseline with Feature Engineering'
    )
    parser.add_argument('--train-file', type=str, default='train.json')
    parser.add_argument('--test-file', type=str, default='test_without_labels.json')
    parser.add_argument('--output-file', type=str, default='submission_enhanced.csv')
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--no-cleaning', action='store_true',
                        help='Disable text cleaning')
    parser.add_argument('--no-stats', action='store_true',
                        help='Disable statistical features')

    args = parser.parse_args()

    use_cleaning = not args.no_cleaning
    use_stats_features = not args.no_stats

    print("\n" + "=" * 60)
    print("PADBen Task1 - Enhanced Baseline")
    print("=" * 60)
    print(f"Train file:           {args.train_file}")
    print(f"Test file:            {args.test_file}")
    print(f"Output file:          {args.output_file}")
    print(f"Max features:         {args.max_features}")
    print(f"N estimators:         {args.n_estimators}")
    print(f"Text cleaning:        {'ON' if use_cleaning else 'OFF'}")
    print(f"Statistical features: {'ON' if use_stats_features else 'OFF'}")

    # 加载数据
    train_sentences, train_labels = load_train_data(args.train_file)
    test_sentences, test_ids = load_test_data(args.test_file)

    # 训练模型
    clf, tfidf_vec, scaler, use_clean, use_stats = train_enhanced_model(
        train_sentences, train_labels,
        max_features=args.max_features,
        n_estimators=args.n_estimators,
        use_cleaning=use_cleaning,
        use_stats_features=use_stats_features
    )

    # 预测
    predict_and_save(clf, tfidf_vec, scaler, test_sentences, test_ids,
                     args.output_file, use_clean, use_stats)

    print("\n" + "=" * 60)
    print("✅ ENHANCED BASELINE COMPLETE!")
    print("=" * 60)
    print(f"\nFeature Engineering Summary:")
    print(f"  • TF-IDF: {args.max_features} dimensions")
    print(f"  • Statistical features: {'8 dimensions' if use_stats else 'Disabled'}")
    print(f"  • Text cleaning: {'Enabled' if use_clean else 'Disabled'}")
    print(f"\nNext steps:")
    print(f"1. Validate: python3 validate_submission.py {args.output_file}")
    print(f"2. Submit to Kaggle")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()