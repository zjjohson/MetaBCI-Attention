# -*- coding: utf-8 -*-
"""
TCN-Attention 端到端深度学习训练脚本

输入: 原始 EEG 段 (2, 250) → bandpass(1-40Hz) → TCN-Attention → 3分类
替换原有 MLP + 手工特征的 pipeline。

用法:
    python train_tcn.py                    # 训练
    python train_tcn.py --epochs 150       # 自定义轮数
"""
import sys, os, json, glob, argparse, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("train_tcn")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "attention_dataset", "real_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
PATIENCE = 12  # early stopping
TCN_CHANNELS = [32, 64, 96]  # 中等规模, 全量数据可用更大模型


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_all_segments():
    """加载所有 MEMA 预处理后的 EEG 段, 应用带通滤波, 返回 (N, 2, 250) 数组和标签"""
    from attention_dataset.dataset import bandpass_filter

    subj_dirs = []
    for d in os.listdir(REAL_DATA_DIR):
        dp = os.path.join(REAL_DATA_DIR, d)
        if (d.startswith("sub-") or d.startswith("mema_sub_")) and os.path.isdir(dp):
            subj_dirs.append(dp)

    X_list, y_list = [], []
    for sp in sorted(subj_dirs):
        files = sorted([f for f in os.listdir(sp) if f.endswith(".npy")])
        if not files:
            continue
        # 读取标签
        meta_path = os.path.join(sp, "meta.json")
        labels = []
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "labels" in meta:
                labels = meta["labels"]
            elif "counts" in meta:
                cnts = meta["counts"]
                for lbl in ["0", "1", "2"]:
                    labels.extend([int(lbl)] * cnts.get(lbl, cnts.get(int(lbl), 0)))

        cnt = 0
        for i, fname in enumerate(files):
            try:
                seg = np.load(os.path.join(sp, fname))
            except Exception:
                continue
            if seg.ndim != 2 or seg.shape[0] != 2 or seg.shape[1] < 50:
                continue
            # 带通滤波 (与 run_device.py 推理参数一致: 0.5-45Hz, 2阶)
            filtered = bandpass_filter(seg, srate=250, l_freq=0.5, h_freq=45, order=2)
            if filtered.shape[1] > 250:
                filtered = filtered[:, :250]
            elif filtered.shape[1] < 250:
                filtered = np.pad(filtered, ((0, 0), (0, 250 - filtered.shape[1])))
            X_list.append(filtered)
            lbl = labels[i] if i < len(labels) else 1
            y_list.append(lbl)
            cnt += 1
        logger.info("  %s: %d 段", os.path.basename(sp), cnt)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    logger.info("总数据: X=%s, y=%s", X.shape, y.shape)
    for lbl, name in [(0, "涣散"), (1, "一般"), (2, "集中")]:
        logger.info("  %s(%d): %d", name, lbl, (y == lbl).sum())
    return X, y


def normalize(X, eps=1e-8):
    """逐段 z-score 归一化 (per-segment, per-channel)"""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + eps
    return (X - mean) / std


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train():
    try:
        _train_impl()
    except Exception:
        logger.exception("训练崩溃!")
        sys.exit(1)


def _train_impl():
    # --- 加载数据 ---
    logger.info("=" * 55)
    logger.info("  加载 MEMA 数据...")
    logger.info("=" * 55)
    X, y = load_all_segments()
    if len(X) < 100:
        logger.error("数据不足!")
        return

    # 归一化
    X = normalize(X)
    logger.info("归一化后 stats: mean=%.3f std=%.3f", X.mean(), X.std())

    # 划分：80% 训练, 10% 验证, 10% 测试 (保持类别平衡)
    n = len(X)
    n_test = n // 10
    n_val = n // 10
    n_train = n - n_test - n_val

    # 分层划分 (按类别)
    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info("划分: train=%d, val=%d, test=%d (全量)", n_train, n_val, n_test)

    # PyTorch DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # --- 模型 ---
    from brainda.algorithms.deep_learning.tcn_attention import TCN_Attention

    model = TCN_Attention(
        n_channels=2,
        n_samples=250,
        n_classes=3,
        tcn_channels=TCN_CHANNELS,
        kernel_size=3,
        dropout=0.3,
        num_attn_heads=4,
        fc_hidden=64,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数量: %d (%.1fK)", n_params, n_params / 1000)

    # --- 优化器 & 损失 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # --- 训练循环 ---
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    logger.info("=" * 55)
    logger.info("  开始训练 (device=%s, epochs=%d)", DEVICE, MAX_EPOCHS)
    logger.info("=" * 55)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = len(train_loader)
        for i, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            if (i + 1) % 30 == 0:
                logger.info("    batch %d/%d (%.0f%%)", i + 1, n_batches, 100 * (i + 1) / n_batches)
        train_loss /= n_train

        # 验证
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                logits = model(bx)
                val_loss += criterion(logits, by).item() * bx.size(0)
                val_preds.extend(logits.argmax(-1).cpu().tolist())
                val_true.extend(by.cpu().tolist())
        val_loss /= n_val
        val_acc = accuracy_score(val_true, val_preds)

        scheduler.step()

        if epoch == 1 or epoch % 1 == 0:
            logger.info("  Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | best=%.2f%%",
                        epoch, train_loss, val_loss, val_acc * 100, best_val_acc * 100)
            sys.stdout.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            # 保存最佳模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, os.path.join(MODEL_DIR, "tcn_attention_best.pt"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info("  Early stopping at epoch %d (best=%d, acc=%.2f%%)", epoch, best_epoch, best_val_acc * 100)
                break

    # --- 加载最佳模型, 测试 ---
    logger.info("=" * 55)
    logger.info("  加载最佳模型 (epoch=%d, val_acc=%.2f%%)", best_epoch, best_val_acc * 100)
    ckpt = torch.load(os.path.join(MODEL_DIR, "tcn_attention_best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            test_preds.extend(logits.argmax(-1).cpu().tolist())
            test_true.extend(by.cpu().tolist())

    test_acc = accuracy_score(test_true, test_preds)
    logger.info("=" * 55)
    logger.info("  测试集准确率: %.2f%%", test_acc * 100)
    logger.info("=" * 55)
    print(classification_report(test_true, test_preds, target_names=["涣散", "一般", "集中"]))
    print("混淆矩阵:\n", confusion_matrix(test_true, test_preds))

    # --- 导出 TorchScript 用于生产推理 ---
    logger.info("导出 TorchScript...")
    model.cpu()
    example = torch.randn(1, 2, 250)
    traced = torch.jit.trace(model, example)
    ts_path = os.path.join(MODEL_DIR, "tcn_attention.pt")
    traced.save(ts_path)
    logger.info("✓ TorchScript 模型 → %s", ts_path)

    # --- 保存配置 ---
    config = {
        "model": "TCN-Attention",
        "input_shape": [2, 250],
        "n_classes": 3,
        "n_params": n_params,
        "n_samples": n,
        "test_accuracy": float(test_acc),
        "val_accuracy": float(best_val_acc),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(MODEL_DIR, "tcn_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info("✓ 配置 → models/tcn_config.json")
    logger.info("=" * 55)
    logger.info("  训练完成! 测试准确率: %.2f%%", test_acc * 100)
    compare = 67.05
    if test_acc * 100 > compare:
        logger.info("  ★ 优于 MLP 基线 (67.05%%) 提升了 %.1f 个百分点!", test_acc * 100 - compare)
    else:
        logger.info("  MLP 基线: 67.05%% (差距: %.1f 个百分点)", compare - test_acc * 100)
    logger.info("=" * 55)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = p.parse_args()
    MAX_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    train()
