# -*- coding: utf-8 -*-
"""
brainflow 实时注意力处理流水线

基于 MetaBCI brainflow.ProcessWorker 的多进程在线 BCI 处理框架。

流水线:
    EEG数据 → 滑窗切片(1.0s) → 带通滤波 → 时频特征提取 → TCN-Attention → Socket发布

支持:
    - 实时数据采集 (Brainflow 统一API / 模拟信号发生器)
    - 1.0s 滑窗(重叠0.5s)实时解码
    - 三分类结果通过 Socket 广播
    - ≤2 导联 (Fp1/Fp2) 配置

Authors: 上大Meta梦
Date: 2026-05
License: MIT
"""
import time
import queue
import threading
import logging
from typing import Optional, Callable

import numpy as np
from scipy import signal as scipy_signal

# brainflow 核心
from metabci.brainflow.workers import ProcessWorker

# 项目自定义模块
from attention_dataset.dataset import (
    extract_time_frequency_features,
    bandpass_filter,
)

logger = logging.getLogger("attention_worker")


class AttentionWorker(ProcessWorker):
    """
    注意力实时解码工作进程

    继承 brainflow ProcessWorker 的多进程框架，负责:
    - pre():   加载模型、初始化 Socket 客户端
    - consume(): 滑窗 → 滤波 → 特征 → 分类 → 发布
    - post():  清理资源

    Parameters
    ----------
    model : object
        预训练的 TCN-Attention 模型 (需实现 predict 方法)
    n_channels : int. 默认 2
    srate : int. 默认 250
    window_size : float. 秒, 默认 1.0
    overlap : float. [0,1), 默认 0.5
    filter_band : tuple. (l_freq, h_freq), 默认 (1, 50)
    feedback_port : int. Socket 端口, 默认 9999

    Examples
    --------
    >>> from brainda.algorithms.deep_learning.tcn_attention import TCN_Attention
    >>> model = TCN_Attention()
    >>> worker = AttentionWorker(model=model, n_channels=2)
    >>> worker.start()
    >>> worker.in_queue.put(eeg_chunk)  # 推送实时 EEG 数据
    """

    def __init__(
        self,
        model,
        n_channels: int = 2,
        srate: int = 250,
        window_size: float = 1.0,
        overlap: float = 0.5,
        filter_band: tuple = (1.0, 50.0),
        feedback_port: int = 9999,
        timeout: float = 0.01,
    ):
        super().__init__(timeout=timeout, name="attention_worker")
        self._model = model
        self.n_channels = n_channels
        self.srate = srate
        self.window_size = window_size
        self.overlap = overlap
        self.filter_band = filter_band
        self.feedback_port = feedback_port

        self._n_samples_window = int(window_size * srate)
        self._step = int(self._n_samples_window * (1 - overlap))

        # 实时缓冲
        self._buffer: list = []
        self._current_state: tuple = (-1, 0.0)

        # Socket 客户端
        self._client = None

        # 统计
        self._state_history: list = []
        self._n_processed = 0
        self._callback: Optional[Callable] = None

    def set_state_callback(self, callback: Callable[[int, float], None]):
        """设置状态更新回调"""
        self._callback = callback

    def pre(self):
        """离线准备：加载模型，建立 Socket 连接"""
        from brainstim.socket_feedback import AttentionFeedbackClient

        logger.info("[AttentionWorker] 预准备阶段: 加载模型...")
        if hasattr(self._model, "eval"):
            self._model.eval()

        logger.info("[AttentionWorker] 连接反馈服务器...")
        self._client = AttentionFeedbackClient(port=self.feedback_port)
        if not self._client.connect():
            logger.warning("[AttentionWorker] 无法连接反馈服务器, 将在本地运行")

        logger.info("[AttentionWorker] 就绪，等待数据...")

    def consume(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        在线处理: 滑窗 → 滤波 → 特征 → 分类

        Parameters
        ----------
        data : np.ndarray
            实时 EEG 数据, shape (n_channels, n_new_samples)

        Returns
        -------
        预测标签或 None
        """
        if data is None:
            return None

        # 追加到缓冲区
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self._buffer.append(data)

        # 拼接缓冲区
        buffer_data = np.concatenate(self._buffer, axis=-1)
        if buffer_data.shape[1] < self._n_samples_window:
            return None

        # 提取窗口
        window = buffer_data[:, : self._n_samples_window]

        # 滑窗: 丢弃已处理样本
        self._buffer = [buffer_data[:, self._step :]] if buffer_data.shape[1] > self._step else []

        # 1. 带通滤波
        filtered = bandpass_filter(
            window,
            srate=self.srate,
            l_freq=self.filter_band[0],
            h_freq=self.filter_band[1],
        )

        # 2. 时频特征 + 模型推理
        features = extract_time_frequency_features(
            filtered[np.newaxis, :, :], srate=self.srate
        )

        # 3. 模型分类
        try:
            if hasattr(self._model, "predict"):
                label = int(self._model.predict(features)[0])
            else:
                # 如果模型直接接受 raw 数据
                import torch
                tensor_input = torch.from_numpy(
                    filtered[np.newaxis, :, :]
                ).float()
                probs = self._model.predict_proba(tensor_input)[0]
                label = int(np.argmax(probs))
                confidence = float(np.max(probs))
        except Exception as e:
            logger.error(f"[AttentionWorker] 推理错误: {e}")
            return None

        confidence = self._get_confidence(filtered, label)

        # 更新状态
        self._current_state = (label, confidence)
        self._state_history.append((label, confidence, time.time()))
        self._n_processed += 1

        # 发布到 Socket
        if self._client:
            try:
                self._client.send_command(f"UPDATE:{label}:{confidence:.3f}")
            except Exception:
                pass

        # 触发回调
        if self._callback:
            try:
                self._callback(label, confidence)
            except Exception:
                pass

        logger.debug(
            f"[AttentionWorker] #{self._n_processed} "
            f"→ 状态: {label}({'涣散/一般/集中'.split('/')[label]}), "
            f"置信度: {confidence:.3f}"
        )

        return np.array([label])

    def _get_confidence(self, data: np.ndarray, label: int) -> float:
        """基于频谱特征估算置信度"""
        try:
            from attention_dataset.dataset import extract_time_frequency_features as ext_feat
            feat = ext_feat(data[np.newaxis, :, :], srate=self.srate)[0]

            # 计算各频带功率比例
            n_ch = self.n_channels
            n_bands = 5
            band_powers = feat[n_ch * 4 : n_ch * 4 + n_ch * n_bands]

            theta_ratio = np.mean(band_powers[1::5])  # theta
            alpha_ratio = np.mean(band_powers[2::5])  # alpha
            beta_ratio = np.mean(band_powers[3::5])  # beta

            if label == 0:  # 涣散: theta 应高
                confidence = min(1.0, theta_ratio * 2)
            elif label == 2:  # 集中: beta 应高
                confidence = min(1.0, beta_ratio * 2)
            else:  # 一般: alpha 应高
                confidence = min(1.0, alpha_ratio * 2)
            return float(confidence)
        except Exception:
            return 0.5

    def post(self):
        """清理: 断开 Socket"""
        logger.info(f"[AttentionWorker] 停止。共处理 {self._n_processed} 个窗口")
        if self._client:
            self._client.close()
        self._state_history.clear()

    def get_current_state(self) -> tuple:
        """获取最新解码状态"""
        return self._current_state


# ---------------------------------------------------------------------------
# 信号发生器 (离线测试用)
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    模拟脑电信号发生器

    用于在没有真实脑电设备时测试整个流水线。
    生成具有类别区分性的合成信号来验证 TCN-Attention 模型。

    Examples
    --------
    >>> gen = SignalGenerator(seed=42)
    >>> worker = AttentionWorker(model=model)
    >>> worker.start()
    >>> for _ in range(100):
    ...     data = gen.next_sample(label=2)  # 模拟"集中"状态
    ...     worker.in_queue.put(data)
    """

    def __init__(
        self,
        n_channels: int = 2,
        srate: int = 250,
        n_samples_per_chunk: int = 50,
        seed: int = 42,
    ):
        self.n_channels = n_channels
        self.srate = srate
        self.n_samples_per_chunk = n_samples_per_chunk
        self.rng = np.random.RandomState(seed)

        # 各频带噪声基底
        self._t = np.arange(n_samples_per_chunk) / srate

    def next_sample(self, label: int) -> np.ndarray:
        """
        生成指定标签的合成 EEG 数据块

        Parameters
        ----------
        label : int
            0=涣散(theta主导), 1=一般(alpha主导), 2=集中(beta主导)
        """
        noise = 0.3 * self.rng.randn(self.n_channels, self.n_samples_per_chunk)
        signal = np.zeros(self.n_samples_per_chunk)

        if label == 0:  # 涣散: 强theta
            for f in self.rng.uniform(4, 7, 3):
                signal += 1.5 * np.sin(2 * np.pi * f * self._t + self.rng.rand())
        elif label == 1:  # 一般: 强alpha
            for f in self.rng.uniform(9, 12, 3):
                signal += 1.5 * np.sin(2 * np.pi * f * self._t + self.rng.rand())
        else:  # 集中: 强beta
            for f in self.rng.uniform(15, 28, 4):
                signal += 1.5 * np.sin(2 * np.pi * f * self._t + self.rng.rand())

        signal = signal.reshape(1, -1)
        data = np.repeat(signal, self.n_channels, axis=0) + noise
        return data
