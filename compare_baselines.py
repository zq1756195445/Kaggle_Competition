#!/usr/bin/env python3
"""
PADBen Task1 - Baseline Comparison Script
自动运行所有baseline并对比结果
"""

import sys
import subprocess
import json
import pandas as pd
import time
import os
from pathlib import Path


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_baseline(script_name, output_file):
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_name, '--output-file', output_file],
            cwd=BASE_DIR,                 # ✅ 确保路径一致
            capture_output=True,
            text=True,
            encoding = "utf-8",  # ✅ 强制 utf-8
            errors = "replace"  # ✅ 遇到坏字符也不要炸
        )
        elapsed = time.time() - start

        stdout_text = result.stdout if result.stdout is not None else ""
        stderr_text = result.stderr if result.stderr is not None else ""

        if result.returncode != 0:
            return {
                "success": False,
                "time": None,
                "stdout": result.stdout,
                "stderr": result.stderr or "Unknown error"
            }

        return {
            "success": True,
            "time": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        return {
            "success": False,
            "time": None,
            "stdout": "",
            "stderr": str(e)
        }



def extract_accuracy(output_text):
    """从输出中提取验证准确率"""
    import re

    # 查找 "Validation Accuracy: 0.8234" 格式
    match = re.search(r'Validation Accuracy:\s*(\d+\.\d+)', output_text)
    if match:
        return float(match.group(1))

    return None


def compare_predictions(files):
    """对比不同baseline的预测结果"""
    print_section("PREDICTION COMPARISON")

    dfs = {}
    for name, filepath in files.items():
        if os.path.exists(filepath):
            dfs[name] = pd.read_csv(filepath)
            print(f"✓ Loaded {name}: {len(dfs[name])} predictions")
        else:
            print(f"✗ Not found: {name}")

    if len(dfs) < 2:
        print("\n⚠️ Need at least 2 files to compare")
        return

    # 统计预测分布
    print("\nPrediction Distribution:")
    print(f"{'Baseline':<25} {'Label 0':<12} {'Label 1':<12}")
    print("-" * 50)

    for name, df in dfs.items():
        counts = df['target'].value_counts().sort_index()
        count_0 = counts.get(0, 0)
        count_1 = counts.get(1, 0)
        pct_0 = count_0 / len(df) * 100
        pct_1 = count_1 / len(df) * 100
        print(f"{name:<25} {count_0:5d} ({pct_0:5.2f}%)  {count_1:5d} ({pct_1:5.2f}%)")

    # 对比一致性
    print("\nPrediction Agreement:")

    names = list(dfs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            df1, df2 = dfs[name1], dfs[name2]

            # 确保ID对齐
            merged = df1.merge(df2, on='id', suffixes=('_1', '_2'))
            agreement = (merged['target_1'] == merged['target_2']).sum()
            total = len(merged)
            pct = agreement / total * 100

            print(f"  {name1} vs {name2}: {agreement}/{total} ({pct:.2f}%) agree")


def main():
    print_section("PADBen Task1 - BASELINE COMPARISON")

    # 检查必要文件
    required_files = ['train.json', 'test_without_labels.json']
    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"   • {f}")
        print("\nPlease download data files from Kaggle first.")
        return

    print("✓ Data files found")

    # 定义要运行的baseline
    baselines = [
        # --- New LR baseline ---
        {
            'name': 'LR (Enhanced Features)',
            'script': 'baseline_lr.py',
            'output': 'submission_lr.csv',
            'skip_if_missing': False
        },

        # --- New SVM baseline ---
        {
            'name': 'SVM (Enhanced Features)',
            'script': 'baseline_svm.py',
            'output': 'submission_svm.csv',
            'skip_if_missing': False
        },

        # --- Existing baselines ---
        {
            'name': 'Random Forest',
            'script': 'baseline_random_forest.py',
            'output': 'submission_rf.csv',
            'skip_if_missing': False
        },
        {
            'name': 'XGBoost',
            'script': 'baseline_xgboost.py',
            'output': 'submission_xgb.csv',
            'skip_if_missing': True
        },
        {
            'name': 'Enhanced (Feature Engineering)',
            'script': 'baseline_enhanced.py',
            'output': 'submission_enhanced.csv',
            'skip_if_missing': False
        }
    ]


    # 运行所有baseline
    print_section("RUNNING BASELINES")

    results = []

    for baseline in baselines:
        name = baseline['name']
        script = baseline['script']
        output = baseline['output']

        # 检查脚本是否存在
        if not os.path.exists(script):
            print(f"⚠️ Skipping {name}: {script} not found")
            continue

        print(f"🚀 Running {name}...")
        print(f"   Script: {script}")
        print(f"   Output: {output}")

        res = run_baseline(script, output)

        if not res["success"]:
            print("❌ Failed!")
            print("Error:", res["stderr"])
            print(res["stdout"])  # ← 就是你要添加的这一行
            print()
            results.append({
                'name': name,
                'success': False,
                'time': None,
                'accuracy': None,
                'output_file': output
            })
            continue

        # 成功时输出
        print(f"✅ Success! Time: {res['time']:.2f}s")

        # 从 stdout 提取验证准确率
        accuracy = extract_accuracy(res["stdout"])

        results.append({
            'name': name,
            'success': True,
            'time': res["time"],
            'accuracy': accuracy,
            'output_file': output
        })
        print()  # 空行

    # 总结结果
    print_section("RESULTS SUMMARY")

    if not results:
        print("❌ No baselines were run successfully")
        return

    # 打印表格
    print(f"{'Baseline':<30} {'Status':<10} {'Time':<12} {'Val Acc':<12} {'Output File'}")
    print("-" * 90)

    for r in results:
        status = "✅ Success" if r['success'] else "❌ Failed"
        time_str = f"{r['time']:.2f}s" if r['success'] else "N/A"
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] else "N/A"

        print(f"{r['name']:<30} {status:<10} {time_str:<12} {acc_str:<12} {r['output_file']}")

    # 对比预测结果
    submission_files = {
        r['name']: r['output_file']
        for r in results
        if r['success'] and os.path.exists(r['output_file'])
    }

    if submission_files:
        compare_predictions(submission_files)

    # 推荐
    print_section("RECOMMENDATIONS")

    successful = [r for r in results if r['success'] and r['accuracy']]

    if successful:
        # 按准确率排序
        successful.sort(key=lambda x: x['accuracy'], reverse=True)

        best = successful[0]
        print(f"🏆 Best baseline: {best['name']}")
        print(f"   Validation Accuracy: {best['accuracy']:.4f}")
        print(f"   Training Time: {best['time']:.2f}s")
        print(f"   Output: {best['output_file']}")

        print(f"\n💡 Recommended for Kaggle submission:")
        print(f"   python3 validate_submission.py {best['output_file']}")
        print(f"   # Then upload {best['output_file']} to Kaggle")
    else:
        print("⚠️ No successful runs with validation accuracy")

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()