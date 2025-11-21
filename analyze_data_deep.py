#!/usr/bin/env python3
"""
PADBen Task1 - 深度数据分析
类似EDA（探索性数据分析），包含可视化和统计
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_data(train_file, test_file=None):
    """加载训练和测试数据"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # 加载训练集（新格式：单句 + 单标签）
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    print(f"\nTraining data: {len(train_data)} sentences")

    # 直接取出 sentence 和 label
    sentences = [item['sentence'] for item in train_data]
    labels = [item['label'] for item in train_data]

    print(f"  → {len(sentences)} sentences after expanding pairs")

    # 加载测试集（如果提供）
    test_sentences = None
    if test_file:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        test_sentences = [item['sentence'] for item in test_data]
        print(f"\nTest data: {len(test_sentences)} sentences")

    return sentences, labels, test_sentences


def analyze_basic_stats(sentences, labels):
    """基本统计分析"""
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)

    # 标签分布
    label_counts = Counter(labels)
    print(f"\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(labels) * 100
        print(f"  Label {label}: {count:5d} ({pct:5.2f}%)")

    # 句子长度统计
    char_lengths = [len(s) for s in sentences]
    word_lengths = [len(s.split()) for s in sentences]

    print(f"\nSentence Length (Characters):")
    print(f"  Mean:   {np.mean(char_lengths):.2f}")
    print(f"  Median: {np.median(char_lengths):.2f}")
    print(f"  Std:    {np.std(char_lengths):.2f}")
    print(f"  Min:    {np.min(char_lengths)}")
    print(f"  Max:    {np.max(char_lengths)}")

    print(f"\nSentence Length (Words):")
    print(f"  Mean:   {np.mean(word_lengths):.2f}")
    print(f"  Median: {np.median(word_lengths):.2f}")
    print(f"  Std:    {np.std(word_lengths):.2f}")
    print(f"  Min:    {np.min(word_lengths)}")
    print(f"  Max:    {np.max(word_lengths)}")

    return char_lengths, word_lengths


def analyze_by_label(sentences, labels):
    """按标签分析"""
    print("\n" + "=" * 60)
    print("ANALYSIS BY LABEL")
    print("=" * 60)

    # 分组
    label_0_sentences = [s for s, l in zip(sentences, labels) if l == 0]
    label_1_sentences = [s for s, l in zip(sentences, labels) if l == 1]

    # 计算统计量
    stats = {}
    for label, sents in [(0, label_0_sentences), (1, label_1_sentences)]:
        word_lengths = [len(s.split()) for s in sents]
        char_lengths = [len(s) for s in sents]

        # 标点符号统计
        punct_counts = [sum(1 for c in s if c in '.,;!?:\'"-') for s in sents]

        # 大写字母统计
        upper_counts = [sum(1 for c in s if c.isupper()) for s in sents]
        upper_ratios = [uc / len(s) if len(s) > 0 else 0
                        for uc, s in zip(upper_counts, sents)]

        stats[label] = {
            'count': len(sents),
            'word_len_mean': np.mean(word_lengths),
            'word_len_std': np.std(word_lengths),
            'char_len_mean': np.mean(char_lengths),
            'char_len_std': np.std(char_lengths),
            'punct_mean': np.mean(punct_counts),
            'upper_ratio_mean': np.mean(upper_ratios)
        }

    # 打印对比
    print(f"\n{'Metric':<25} {'Label 0':<15} {'Label 1':<15}")
    print("-" * 55)
    print(f"{'Count':<25} {stats[0]['count']:<15d} {stats[1]['count']:<15d}")
    print(f"{'Word Length (mean)':<25} {stats[0]['word_len_mean']:<15.2f} {stats[1]['word_len_mean']:<15.2f}")
    print(f"{'Word Length (std)':<25} {stats[0]['word_len_std']:<15.2f} {stats[1]['word_len_std']:<15.2f}")
    print(f"{'Char Length (mean)':<25} {stats[0]['char_len_mean']:<15.2f} {stats[1]['char_len_mean']:<15.2f}")
    print(f"{'Punctuation (mean)':<25} {stats[0]['punct_mean']:<15.2f} {stats[1]['punct_mean']:<15.2f}")
    print(f"{'Upper Ratio (mean)':<25} {stats[0]['upper_ratio_mean']:<15.4f} {stats[1]['upper_ratio_mean']:<15.4f}")

    return stats


def visualize_distributions(sentences, labels, save_path='analysis_plots.png'):
    """可视化数据分布"""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 准备数据
    label_0_sentences = [s for s, l in zip(sentences, labels) if l == 0]
    label_1_sentences = [s for s, l in zip(sentences, labels) if l == 1]

    label_0_word_lens = [len(s.split()) for s in label_0_sentences]
    label_1_word_lens = [len(s.split()) for s in label_1_sentences]

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 标签分布
    ax1 = axes[0, 0]
    label_counts = Counter(labels)
    colors = ['#4e79a7', '#f28e2b']
    ax1.bar(label_counts.keys(), label_counts.values(), color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Label', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Label Distribution', fontsize=14, pad=15)
    ax1.set_xticks([0, 1])
    for i, (label, count) in enumerate(sorted(label_counts.items())):
        pct = count / len(labels) * 100
        ax1.text(label, count + 50, f'{count}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    # 2. 词长度分布（按标签）
    ax2 = axes[0, 1]
    ax2.hist([label_0_word_lens, label_1_word_lens], bins=30,
             label=['Label 0', 'Label 1'], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Word Count', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Word Count Distribution by Label', fontsize=14, pad=15)
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', alpha=0.3)

    # 3. 词长度箱线图（按标签）
    ax3 = axes[1, 0]
    box_data = [label_0_word_lens, label_1_word_lens]
    bp = ax3.boxplot(box_data, labels=['Label 0', 'Label 1'],
                     patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Word Count', fontsize=12)
    ax3.set_title('Word Count Distribution (Boxplot)', fontsize=14, pad=15)
    ax3.grid(True, axis='y', alpha=0.3)

    # 4. 统计特征对比
    ax4 = axes[1, 1]

    # 计算各种统计量
    label_0_punct = [sum(1 for c in s if c in '.,;!?:\'"-') for s in label_0_sentences]
    label_1_punct = [sum(1 for c in s if c in '.,;!?:\'"-') for s in label_1_sentences]

    label_0_upper = [sum(1 for c in s if c.isupper()) / len(s) if len(s) > 0 else 0
                     for s in label_0_sentences]
    label_1_upper = [sum(1 for c in s if c.isupper()) / len(s) if len(s) > 0 else 0
                     for s in label_1_sentences]

    metrics = ['Word Count', 'Punctuation', 'Upper Ratio']
    label_0_means = [
        np.mean(label_0_word_lens),
        np.mean(label_0_punct),
        np.mean(label_0_upper) * 100  # 转为百分比
    ]
    label_1_means = [
        np.mean(label_1_word_lens),
        np.mean(label_1_punct),
        np.mean(label_1_upper) * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, label_0_means, width, label='Label 0',
                    color=colors[0], alpha=0.8, edgecolor='black')
    bars2 = ax4.bar(x + width / 2, label_1_means, width, label='Label 1',
                    color=colors[1], alpha=0.8, edgecolor='black')

    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Statistical Features Comparison', fontsize=14, pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")

    return fig


def analyze_vocabulary(sentences, labels):
    """词汇分析"""
    print("\n" + "=" * 60)
    print("VOCABULARY ANALYSIS")
    print("=" * 60)

    # 全部词汇
    all_words = []
    for s in sentences:
        all_words.extend(s.lower().split())

    vocab_size = len(set(all_words))
    total_words = len(all_words)

    print(f"\nOverall Vocabulary:")
    print(f"  Unique words: {vocab_size:,}")
    print(f"  Total words:  {total_words:,}")
    print(f"  Avg per sentence: {total_words / len(sentences):.2f}")

    # 最常见的词
    word_counts = Counter(all_words)
    print(f"\nTop 20 Most Common Words:")
    for word, count in word_counts.most_common(20):
        pct = count / total_words * 100
        print(f"  {word:15s} {count:6d} ({pct:5.2f}%)")

    # 按标签分析
    label_0_words = []
    label_1_words = []
    for s, l in zip(sentences, labels):
        words = s.lower().split()
        if l == 0:
            label_0_words.extend(words)
        else:
            label_1_words.extend(words)

    print(f"\nVocabulary by Label:")
    print(f"  Label 0: {len(set(label_0_words)):,} unique words")
    print(f"  Label 1: {len(set(label_1_words)):,} unique words")

    # 标签特有词汇
    label_0_set = set(label_0_words)
    label_1_set = set(label_1_words)

    only_0 = label_0_set - label_1_set
    only_1 = label_1_set - label_0_set
    shared = label_0_set & label_1_set

    print(f"  Label 0 only: {len(only_0):,} words")
    print(f"  Label 1 only: {len(only_1):,} words")
    print(f"  Shared:       {len(shared):,} words")


def generate_report(train_file, test_file=None, output_plot='analysis_plots.png'):
    """生成完整分析报告"""
    print("\n" + "=" * 60)
    print("PADBen Task1 - DATA ANALYSIS REPORT")
    print("=" * 60)

    # 加载数据
    sentences, labels, test_sentences = load_data(train_file, test_file)

    # 基本统计
    char_lengths, word_lengths = analyze_basic_stats(sentences, labels)

    # 按标签分析
    stats = analyze_by_label(sentences, labels)

    # 词汇分析
    analyze_vocabulary(sentences, labels)

    # 可视化
    fig = visualize_distributions(sentences, labels, output_plot)

    # 测试集分析
    if test_sentences:
        print("\n" + "=" * 60)
        print("TEST SET ANALYSIS")
        print("=" * 60)

        test_word_lens = [len(s.split()) for s in test_sentences]
        print(f"\nTest Sentence Length (Words):")
        print(f"  Mean:   {np.mean(test_word_lens):.2f}")
        print(f"  Median: {np.median(test_word_lens):.2f}")
        print(f"  Std:    {np.std(test_word_lens):.2f}")
        print(f"  Min:    {np.min(test_word_lens)}")
        print(f"  Max:    {np.max(test_word_lens)}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  • Visualization: {output_plot}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='PADBen Task1 - Deep Data Analysis'
    )
    parser.add_argument('--train-file', type=str, default='train.json',
                        help='Path to training data')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Path to test data (optional)')
    parser.add_argument('--output-plot', type=str, default='analysis_plots.png',
                        help='Path to save visualization')

    args = parser.parse_args()

    generate_report(args.train_file, args.test_file, args.output_plot)


if __name__ == "__main__":
    main()