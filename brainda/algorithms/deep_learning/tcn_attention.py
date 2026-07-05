# -*- coding: utf-8 -*-
"""
TCN-Attention 时域注意力分类模型

面向极稀疏导联（≤2 导联, Fp1/Fp2）的注意力时序深度学习解码网络。
通过时间卷积网络（TCN）捕获时域依赖，结合自注意力机制弥补空间特征缺失。

架构:
    Input: (B, 2, 250)  — 2 导联 × 250 采样点 (1.0s @ 250Hz)
    ├── TemporalConv Block × 3 (dilated causal conv)
    ├── Multi-Head Self-Attention
    ├── Global Avg Pooling
    └── FC → 3 类输出 (涣散/一般/集中)

Authors: 上大Meta梦
Date: 2026-05
License: MIT
"""

from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """因果膨胀卷积 (保持时序因果性)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        # 去掉因果填充的尾部
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    TCN 时序块: 双层因果膨胀卷积 + 残差连接 + 权重归一化 + ReLU + Dropout

    参考: Bai et al. "An Empirical Evaluation of Generic Convolutional
           and Recurrent Networks for Sequence Modeling" (2018)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 残差连接: 当维度不匹配时用 1x1 conv 对齐
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + residual)


class MultiHeadSelfAttention1D(nn.Module):
    """
    1D 时序多头自注意力

    在时间维度上应用自注意力，帮助极少导联下补足空间信息缺失
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)  — batch, seq_len, embed_dim
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, L, D_h)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 主模型: TCN-Attention
# ---------------------------------------------------------------------------


class TCN_Attention(nn.Module):
    """
    TCN-Attention: 面向极稀疏导联的注意力时序深度学习解码网络

    Parameters
    ----------
    n_channels : int
        导联数，默认 2 (Fp1, Fp2)
    n_samples : int
        单窗采样点数，默认 250 (1.0s @ 250Hz)
    n_classes : int
        分类数，默认 3 (涣散/一般/集中)
    tcn_channels : list
        各层 TCN 输出通道，默认 [32, 64, 128]
    kernel_size : int
        TCN 卷积核大小
    dropout : float
        Dropout 比例
    num_attn_heads : int
        自注意力头数
    fc_hidden : int
        FC 隐藏层维度

    Examples
    --------
    >>> model = TCN_Attention(n_channels=2, n_samples=250, n_classes=3)
    >>> x = torch.randn(16, 2, 250)  # (batch, channels, samples)
    >>> out = model(x)
    >>> print(out.shape)  # (16, 3)
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_samples: int = 250,
        n_classes: int = 3,
        tcn_channels: Optional[list] = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        num_attn_heads: int = 4,
        fc_hidden: int = 64,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [32, 64, 128]

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # --- 空间映射层: 将导联维度映射到特征维度 ---
        self.spatial_proj = nn.Conv1d(n_channels, tcn_channels[0], kernel_size=1)

        # --- TCN 模块 (多层因果膨胀卷积) ---
        tcn_blocks = []
        for i in range(len(tcn_channels)):
            in_ch = tcn_channels[i - 1] if i > 0 else tcn_channels[0]
            out_ch = tcn_channels[i]
            dilation = 2 ** i
            tcn_blocks.append(
                (
                    f"temporal_block_{i}",
                    TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout),
                )
            )
        self.tcn = nn.Sequential(OrderedDict(tcn_blocks))

        # --- 自注意力模块 ---
        self.attn = MultiHeadSelfAttention1D(
            embed_dim=tcn_channels[-1],
            num_heads=num_attn_heads,
            dropout=dropout,
        )

        # --- 分类头 ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T) = (batch, 2, 250)
        x = self.spatial_proj(x)  # (B, 32, 250)
        x = self.tcn(x)  # (B, 128, 250)

        # 转置为 (B, L, D) 供 self-attention 使用
        x_t = x.transpose(1, 2)  # (B, 250, 128)
        x_t = self.attn(x_t)  # (B, 250, 128)

        # 回到 (B, D, L) 做全局池化
        x = x_t.transpose(1, 2) + x  # 残差连接
        x = self.global_pool(x).squeeze(-1)  # (B, 128)
        return self.classifier(x)  # (B, 3)

    def predict_proba(self, x: Tensor) -> np.ndarray:
        """返回 softmax 概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict(self, x: Tensor) -> np.ndarray:
        """返回预测类别"""
        return np.argmax(self.predict_proba(x), axis=-1)


# ---------------------------------------------------------------------------
# Skorch 兼容包装器 (兼容 brainda 训练工具链)
# ---------------------------------------------------------------------------

try:
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping, LRScheduler
    from metabci.brainda.algorithms.deep_learning.base import SkorchNet

    class SkorchTCN_Attention(SkorchNet):
        """
        Skorch 兼容的 TCN-Attention，可直接用于 brainda 的 sklearn 风格训练流程

        Examples
        --------
        >>> from sklearn.model_selection import cross_val_score
        >>> net = SkorchTCN_Attention(module__n_channels=2)
        >>> net.fit(X_train, y_train)
        >>> scores = cross_val_score(net, X, y, cv=5)
        """

        def __init__(
            self,
            module__n_channels: int = 2,
            module__n_samples: int = 250,
            module__n_classes: int = 3,
            module__tcn_channels: Optional[list] = None,
            module__kernel_size: int = 3,
            module__dropout: float = 0.3,
            module__num_attn_heads: int = 4,
            module__fc_hidden: int = 64,
            max_epochs: int = 100,
            lr: float = 1e-3,
            batch_size: int = 32,
            **kwargs,
        ):
            super().__init__(
                TCN_Attention,
                module__n_channels=module__n_channels,
                module__n_samples=module__n_samples,
                module__n_classes=module__n_classes,
                module__tcn_channels=module__tcn_channels,
                module__kernel_size=module__kernel_size,
                module__dropout=module__dropout,
                module__num_attn_heads=module__num_attn_heads,
                module__fc_hidden=module__fc_hidden,
                max_epochs=max_epochs,
                lr=lr,
                batch_size=batch_size,
                **kwargs,
            )

except ImportError:
    class SkorchTCN_Attention:
        """占位：当 skorch 不可用时"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "skorch 未安装。运行: pip install skorch"
            )


# ---------------------------------------------------------------------------
# 模型导出工具
# ---------------------------------------------------------------------------

def export_for_inference(
    model: TCN_Attention,
    save_path: str,
    example_input: Optional[Tensor] = None,
):
    """
    导出模型为 TorchScript 用于生产部署

    Parameters
    ----------
    model : TCN_Attention
    save_path : str
    example_input : Tensor, shape (1, 2, 250)
    """
    if example_input is None:
        example_input = torch.randn(1, model.n_channels, model.n_samples)
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(save_path)
    print(f"[TCN-Attention] 模型已导出至: {save_path}")


def load_inference_model(model_path: str) -> torch.jit.ScriptModule:
    """加载 TorchScript 模型"""
    return torch.jit.load(model_path)
