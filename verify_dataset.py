# -*- coding: utf-8 -*-
"""
数据集验证脚本 — 检查 MEMA 预处理和个人采集数据的完整性

用法:
    python verify_dataset.py               # 验证所有数据
    python verify_dataset.py --subject 1   # 只验证 sub-01
"""
import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "attention_dataset", "real_data")


def verify_dir(dirpath):
    """验证单个数据目录, 返回 (文件数, 分布, 错误列表)"""
    name = os.path.basename(dirpath)
    files = sorted([f for f in os.listdir(dirpath) if f.endswith(".npy")])
    meta_path = os.path.join(dirpath, "meta.json")
    errors = []
    labels = []

    # 读标签
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "labels" in meta:
            labels = meta["labels"]
        elif "counts" in meta:
            for lbl in [0, 1, 2]:
                k = str(lbl) if str(lbl) in meta["counts"] else lbl
                labels.extend([lbl] * meta["counts"].get(k, 0))
    labels = labels[:len(files)]

    # 逐文件检查
    for i, fname in enumerate(files):
        fpath = os.path.join(dirpath, fname)
        try:
            arr = np.load(fpath)
        except Exception as e:
            errors.append(f"{fname}: 读取失败 ({e})")
            continue
        if arr.ndim != 2:
            errors.append(f"{fname}: 维度错误 (期望2, 实际{arr.ndim})")
        elif arr.shape[0] != 2:
            errors.append(f"{fname}: 通道数错误 (期望2, 实际{arr.shape[0]})")
        elif arr.shape[1] != 250:
            errors.append(f"{fname}: 样本数错误 (期望250, 实际{arr.shape[1]})")

    dist = {0: 0, 1: 0, 2: 0}
    for i in range(len(files)):
        if i < len(labels):
            dist[labels[i]] = dist.get(labels[i], 0) + 1
        else:
            dist[-1] = dist.get(-1, 0) + 1

    return len(files), dist, errors, sum(labels) if labels else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=str, default=None, help="指定受试者编号, 如 1 或 mema_sub_00")
    args = p.parse_args()

    print("=" * 55)
    print("  数据集完整性验证")
    print("=" * 55)

    if args.subject:
        dirs = [f"sub-{str(args.subject).zfill(2)}" if args.subject.isdigit()
                else args.subject]
    else:
        dirs = sorted([d for d in os.listdir(REAL_DATA_DIR)
                       if os.path.isdir(os.path.join(REAL_DATA_DIR, d))])

    total_files, total_errors = 0, 0
    for d in dirs:
        dp = os.path.join(REAL_DATA_DIR, d)
        if not os.path.isdir(dp):
            continue
        n, dist, errors, _ = verify_dir(dp)
        total_files += n
        total_errors += len(errors)
        status = "OK" if not errors else f"ERROR {len(errors)}个"
        label_names = {0: "涣散", 1: "一般", 2: "集中"}
        dist_str = ", ".join(f"{label_names.get(k, k)}:{v}" for k, v in sorted(dist.items()))
        print(f"  {d}: {n}文件 [{dist_str}] {status}")
        for e in errors[:5]:
            print(f"    * {e}")
        if len(errors) > 5:
            print(f"    ... 及另外 {len(errors)-5} 个错误")

    print("-" * 55)
    print(f"  总计: {total_files} 文件, {total_errors} 错误")
    if total_errors:
        print("  [WARN] 数据存在问题, 请重新运行预处理或采集")
    else:
        print("  [OK] 数据格式完整, 可以用于训练")

    # 检查模型文件
    print("-" * 55)
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    expected_models = [
        "tcn_attention.pt", "tcn_attention_best.pt", "tcn_config.json",
        "real_classifier.pkl", "real_scaler.pkl",
    ]
    for m in expected_models:
        mp = os.path.join(model_dir, m)
        if os.path.exists(mp):
            size = os.path.getsize(mp)
            print(f"  [OK] {m} ({size/1024:.0f}KB)")
        else:
            print(f"  [MISS] {m} - 缺失!")
    print("=" * 55)


if __name__ == "__main__":
    main()
