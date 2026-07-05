# -*- coding: utf-8 -*-
"""
MEMA 公开数据集下载 + 预处理脚本

从 label_attention.mat 读取真实标签 (0=涣散, 1=一般, 2=集中)
从 SubjectX.mat 读取 32 通道 EEG → 提取 Fp1/Fp2 (前2通道)

用法:
    python download_mema.py              # 预处理
    python collect_real_data.py --train  # 训练

MEMA 数据下载:
    链接: https://pan.baidu.com/s/1ssvZWAI6gwV2ey0cRogDWg?pwd=2dg7
    只下载 For_DL 目录, 解压到 attention_dataset/mema/For_DL/
"""
import sys, os, subprocess, glob, json, argparse, logging, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("mema")

MEMA_DIR = os.path.join(os.path.dirname(__file__), "attention_dataset", "mema")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "attention_dataset", "real_data", "mema_sub_00")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess():
    import scipy.io as sio
    logger.info("=" * 55)
    logger.info("  MEMA 数据集预处理 (使用 label_attention.mat)")
    logger.info("=" * 55)

    # 1. 加载标签
    label_path = os.path.join(MEMA_DIR, "For_DL", "label_attention.mat")
    if not os.path.exists(label_path):
        logger.error(f"未找到 {label_path}")
        return
    label_mat = sio.loadmat(label_path)["label"]  # (20, 12)
    logger.info(f"标签矩阵: {label_mat.shape}, 值={np.unique(label_mat)}")
    # 验证映射: 0=涣散, 1=一般, 2=集中 (与MEMA一致)

    # 2. 找受试者文件
    subject_files = sorted(glob.glob(os.path.join(MEMA_DIR, "For_DL", "Subject*.mat")))
    logger.info(f"找到 {len(subject_files)} 个受试者文件")

    trial_idx = 0
    all_labels = []

    for fpath in subject_files:
        m = re.search(r"Subject(\d+)", os.path.basename(fpath))
        if not m:
            continue
        subj = int(m.group(1)) - 1

        try:
            mat = sio.loadmat(fpath)
            # 找 2D EEG array (32+ 通道)
            eeg = None
            for k in mat:
                if isinstance(mat[k], np.ndarray) and mat[k].ndim >= 2 and mat[k].shape[0] >= 30:
                    eeg = mat[k]
                    break
            if eeg is None:
                logger.warning(f"Subject{subj+1}: 未找到EEG数据, keys={list(mat.keys())[:5]}")
                continue

            # 只取前2通道 (Fp1/Fp2)
            eeg = eeg[:2, :].astype(np.float64)
            total_pts = eeg.shape[1]
            n_trials = label_mat.shape[1]  # 12
            trial_len = total_pts // n_trials

            for t in range(n_trials):
                lbl = int(label_mat[subj, t])
                s0, s1 = t * trial_len, (t + 1) * trial_len
                td = eeg[:, s0:s1]

                # 1秒窗口, 50%重叠
                for start in range(0, td.shape[1] - 250, 125):
                    seg = td[:, start:start + 250]
                    np.save(os.path.join(OUTPUT_DIR, f"trial_{trial_idx:04d}.npy"), seg)
                    all_labels.append(lbl)
                    trial_idx += 1

            logger.info(f"  Subject{subj+1}: +{n_trials} trials")

        except Exception as e:
            logger.error(f"Subject{subj+1} 失败: {e}")

    meta = {
        "source": "MEMA",
        "counts": {str(i): all_labels.count(i) for i in [0, 1, 2]},
        "total": trial_idx,
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 55)
    logger.info(f"  完成! 总段数: {trial_idx}")
    for i, name in [(0, "涣散"), (1, "一般"), (2, "集中")]:
        logger.info(f"  {name}({i}): {all_labels.count(i)}")
    logger.info(f"  下一步: python collect_real_data.py --train")
    logger.info("=" * 55)


if __name__ == "__main__":
    if not os.path.exists(label_path := os.path.join(MEMA_DIR, "For_DL")):
        logger.error("For_DL 目录不存在。请从百度网盘下载 MEMA/For_DL 并解压到: %s", MEMA_DIR)
        sys.exit(1)
    preprocess()
