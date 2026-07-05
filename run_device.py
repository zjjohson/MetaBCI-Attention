# -*- coding: utf-8 -*-
"""
真实脑电设备对接模块

用法:
    python run_device.py --device simulate      # 模拟模式
    python run_device.py --device neuracle_tcp  # Neuracle TCP 127.0.0.1:8712
"""
import sys, os, time, logging, threading, argparse
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("device_runner")


class DeviceAdapter:
    def __init__(self, n_channels: int, srate: int):
        self.n_channels = n_channels
        self.srate = srate
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        raise NotImplementedError

    def get_data(self, n_samples: int = 50) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


# ===== LSL =====
class LSLDeviceAdapter(DeviceAdapter):
    def __init__(self, n_channels=2, srate=250, stream_name="", source_id="", timeout=10.0):
        super().__init__(n_channels, srate)
        self.stream_name, self.source_id, self.timeout = stream_name, source_id, timeout
        self._inlet, self._eeg_ch = None, None

    def connect(self) -> bool:
        try:
            from pylsl import resolve_byprop, StreamInlet
            if self.stream_name:
                streams = resolve_byprop("name", self.stream_name, timeout=self.timeout)
            elif self.source_id:
                streams = resolve_byprop("source_id", self.source_id, timeout=self.timeout)
            else:
                streams = resolve_byprop("type", "EEG", timeout=self.timeout)
            if not streams:
                logger.error("未找到 LSL 流"); return False
            s = streams[0]
            self._inlet = StreamInlet(s)
            self._eeg_ch = list(range(min(self.n_channels, s.channel_count())))
            self._connected = True
            logger.info(f"LSL: {s.name()} ({s.channel_count()}ch)")
            return True
        except Exception as e:
            logger.error(f"LSL 失败: {e}"); return False

    def get_data(self, n_samples=50):
        if not self._connected or not self._inlet: return None
        try:
            samples, _ = self._inlet.pull_chunk(max_samples=n_samples, timeout=0.0)
            if not samples: return None
            arr = np.array(samples)
            if self._eeg_ch: arr = arr[:, self._eeg_ch]
            if arr.shape[0] < n_samples:
                arr = np.vstack([arr, np.zeros((n_samples - arr.shape[0], arr.shape[1]))])
            elif arr.shape[0] > n_samples: arr = arr[:n_samples, :]
            return arr.T.astype(np.float64)
        except Exception: return None

    def close(self):
        if self._inlet:
            try: self._inlet.close_stream()
            except: pass
        self._connected = False


# ===== Neuracle TCP =====
class NeuracleTCPAdapter(DeviceAdapter):
    """Neuracle TCP 直连 (127.0.0.1:8712)
    读取 float32 流 → 提取 Fp1/Fp2 → 可选降采样 → ADC→μV"""
    def __init__(self, n_channels=2, srate=250, host="127.0.0.1", port=8712,
                 tcp_channels=64, ch_fp1=0, ch_fp2=1, decimate=1, adc_gain=1.0):
        super().__init__(n_channels, srate)
        self.host, self.port = host, port
        self.tcp_channels = tcp_channels
        self.ch_fp1 = ch_fp1
        self.ch_fp2 = ch_fp2
        self.decimate = decimate  # 降采样因子 (4=1000→250Hz, 1=不降采样)
        self.adc_gain = adc_gain  # ADC→μV 转换系数 (典型值 0.02235)
        self._sock = None
        self._get_cnt = 0
        self._overflow = None
        self._raw_buf = b""  # ★ 原始字节缓冲, 不丢弃
        self._raw_extra = None  # 降采样尾数 (1000→250Hz)
        self._diag = {"samples_out": 0, "last_log": time.time()}  # 诊断

    def connect(self) -> bool:
        import socket
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(0.001)  # 非阻塞: 有数据就读, 没数据立即返回
            self._get_cnt = 0
            self._overflow = None
            self._raw_extra = None
            self._raw_buf = b""
            logger.info(f"Neuracle TCP 已连接 {self.host}:{self.port}")
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"TCP连接失败: {e}")
            return False

    def get_data(self, n_samples=50) -> Optional[np.ndarray]:
        if not self._connected or not self._sock:
            return None
        import socket
        tcp_ch = self.tcp_channels
        try:
            # 1) 读取原始字节到缓冲区
            try:
                chunk = self._sock.recv(65536)
                if chunk:
                    self._raw_buf += chunk
            except socket.timeout:
                pass  # 没读到新数据, 继续用缓冲区

            # 2) 从字节缓冲区解析 float32 → reshape 64通道 → 提取 Fp1/Fp2
            bytes_per_sample = tcp_ch * 4  # 64 × 4 = 256
            usable = (len(self._raw_buf) // bytes_per_sample) * bytes_per_sample
            if usable == 0:
                return None

            raw_data = self._raw_buf[:usable]
            self._raw_buf = self._raw_buf[usable:]  # 保留剩余字节

            # 首次诊断: 打印原始hex
            if self._get_cnt == 0 and len(raw_data) >= 32:
                hex_str = ' '.join(f'{b:02x}' for b in raw_data[:32])
                logger.info(f"[字节诊断] 首包前32B: {hex_str}")

            arr = np.frombuffer(raw_data, dtype=np.float32)
            arr = arr.reshape(tcp_ch, -1).astype(np.float64)
            arr = np.stack([arr[self.ch_fp1, :], arr[self.ch_fp2, :]])  # (2, n) @原始采样率
            arr = arr * self.adc_gain  # ADC→μV

            # ★ 降采样 (decimate倍, 余数跨调用累积)
            dec = self.decimate
            if dec > 1:
                if self._raw_extra is not None:
                    arr = np.concatenate([self._raw_extra, arr], axis=-1)
                    self._raw_extra = None
                rem = arr.shape[1] % dec
                if rem > 0:
                    self._raw_extra = arr[:, -rem:]
                    arr = arr[:, :-rem]
                if arr.shape[1] < dec:
                    return None
                arr = arr[:, ::dec]

            if self._get_cnt == 0:
                self._get_cnt += 1
                logger.info(f"[OK] {tcp_ch}ch→Fp1(ch{self.ch_fp1+1})/Fp2(ch{self.ch_fp2+1}) "
                            f"×{dec}, n={arr.shape[1]} 值≈{arr[0,0]:.0f}/{arr[1,0]:.0f}")

            # 3) 拼接到溢出缓冲, 返回 n_samples
            if self._overflow is not None:
                arr = np.concatenate([self._overflow, arr], axis=-1)
                self._overflow = None

            if arr.shape[1] < n_samples:
                self._overflow = arr
                return None
            else:
                result = arr[:, :n_samples]  # 已在前面完成 ADC→μV
                self._overflow = (arr[:, n_samples:]
                                  if arr.shape[1] > n_samples else None)
                # 诊断: 每5秒输出一次数据速率
                self._diag["samples_out"] += result.shape[1]
                now = time.time()
                if now - self._diag["last_log"] >= 5:
                    logger.info(f"[数据速率] {self._diag['samples_out']/5:.0f} samples/s "
                                f"@250Hz (≈{self._diag['samples_out']/5/250:.1f}x 实时)")
                    self._diag["samples_out"] = 0
                    self._diag["last_log"] = now
                return result

        except Exception as e:
            self._get_cnt += 1
            if self._get_cnt <= 3:
                logger.error(f"[Neuracle错误] get_data异常: {e}")
            return None

    def close(self):
        self._connected = False
        if self._sock:
            try: self._sock.close()
            except: pass


# ===== MetaBCI LSL =====
class MetaBCILSLAdapter(DeviceAdapter):
    def __init__(self, n_channels=2, srate=250, stream_name="", timeout=10.0):
        super().__init__(n_channels, srate)
        self.stream_name, self.timeout = stream_name, timeout
        self._inlet = None

    def connect(self) -> bool:
        try:
            import pylsl
            from metabci.brainflow.amplifiers import DataInlet
            streams = pylsl.resolve_byprop("name", self.stream_name, timeout=self.timeout) if self.stream_name else pylsl.resolve_byprop("type", "EEG", timeout=self.timeout)
            if not streams: logger.error("未找到 LSL 流"); return False
            self._inlet = DataInlet(streams[0])
            self._connected = True; return True
        except Exception as e: logger.error(f"MetaBCI LSL 失败: {e}"); return False

    def get_data(self, n_samples=50):
        if not self._connected or not self._inlet: return None
        try:
            self._inlet.stream_action(); data = self._inlet.get_data()
            if data is None or (isinstance(data, np.ndarray) and data.size <= 1): return None
            arr = np.atleast_2d(data)
            n_ch = min(self.n_channels, arr.shape[1] if arr.ndim > 1 else 1)
            arr = arr[:, :n_ch]
            if arr.shape[0] < n_samples:
                arr = np.vstack([arr, np.zeros((n_samples - arr.shape[0], n_ch))])
            elif arr.shape[0] > n_samples: arr = arr[:n_samples, :]
            return arr.T.astype(np.float64)
        except Exception: return None

    def close(self):
        self._connected = False


# ===== 模拟 =====
class SimulatedDeviceAdapter(DeviceAdapter):
    def __init__(self, n_channels=2, srate=250, seed=42):
        super().__init__(n_channels, srate)
        from brainflow.attention_worker import SignalGenerator
        self._gen = SignalGenerator(n_channels=n_channels, srate=srate, n_samples_per_chunk=50, seed=seed)
        self._labels = [2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 1, 0, 2, 2, 2]
        self._i = 0

    def connect(self) -> bool:
        self._connected = True; logger.info("[模拟] 就绪"); return True

    def get_data(self, n_samples=50):
        if not self._connected: return None
        lbl = self._labels[self._i % len(self._labels)]; self._i += 1
        return self._gen.next_sample(lbl)

    def close(self):
        self._connected = False


# ===== DeviceRunner =====
class DeviceRunner:
    def __init__(self, device, model, scaler, window_size=1.0, srate=250,
                 feedback_port=9999, skip_ms=20, device_type="unknown", model_type="mlp"):
        self.device, self.model, self.scaler = device, model, scaler
        self.srate, self.feedback_port, self.skip_ms = srate, feedback_port, skip_ms
        self.window_samples = int(window_size * srate)
        self.step_samples = int(self.window_samples * 0.5)
        self.device_type = device_type
        self.model_type = model_type
        self._buffer, self._running, self._n = [], False, 0
        self._server, self._controller = None, None

    def start(self):
        from brainstim.socket_feedback import AttentionFeedbackServer, AdaptiveController
        from attention_dataset.dataset import extract_time_frequency_features, bandpass_filter
        if not self.device.connect(): logger.error("设备连接失败!"); return
        self._server = AttentionFeedbackServer(port=self.feedback_port, device_type=self.device_type)
        self._server.start()
        self._controller = AdaptiveController(min_confidence=0.4, cooldown_seconds=2.0)
        labels = {0: "涣散", 1: "一般", 2: "集中"}
        def _alert_beep():
            """播放短促提示音 (不阻塞主线程)"""
            import threading
            def _beep():
                try:
                    import winsound
                    winsound.Beep(800, 300)  # 800Hz, 300ms
                except Exception:
                    pass
            threading.Thread(target=_beep, daemon=True).start()

        self._controller.register_actions(
            on_focused=lambda: logger.info("  🔔 自适应: 集中 -> 屏蔽干扰"),
            on_distracted=lambda: (_alert_beep(), logger.info("  🔔 自适应: 涣散 -> 白噪音提醒")),
            on_normal=lambda: logger.info("  🔔 自适应: 一般 -> 保持当前"),
        )
        logger.info("=" * 50)
        logger.info(f"  实时处理流水线已启动 ({type(self.device).__name__})")
        logger.info(f"  窗口: {self.window_samples / self.srate:.1f}s, Socket: 9999")
        logger.info("=" * 50)
        self._running = True
        wave_path = os.path.join(os.path.dirname(__file__), '.eeg_wave.txt')
        SCALE = 0.001
        _wave_seq = 0  # 波形序列号, 前端去重

        try:
            while self._running:
                data = self.device.get_data(n_samples=self.step_samples)
                if data is None:
                    time.sleep(self.skip_ms / 1000)
                    continue

                self._buffer.append(data)
                buf = np.concatenate(self._buffer, axis=-1)
                if buf.shape[1] >= self.window_samples:
                    win = buf[:, :self.window_samples]
                    self._buffer = [buf[:, self.step_samples:]]
                    filtered = bandpass_filter(win, srate=self.srate,
                                               l_freq=0.5, h_freq=45, order=2)
                    f_mean = filtered.mean(axis=-1, keepdims=True)
                    f_std = filtered.std(axis=-1, keepdims=True) + 1e-8
                    filtered_norm = (filtered - f_mean) / f_std

                    # ★ 写 α波(8-13Hz) 和 β波(14-30Hz)
                    try:
                        # 用 Fp1 通道提取频带 (数据已在适配器降采样到250Hz)
                        sig = win[0, :self.window_samples].astype(np.float64)
                        # α 波 8-13Hz
                        alpha = bandpass_filter(
                            sig[np.newaxis, :], srate=self.srate,
                            l_freq=8, h_freq=13, order=3)[0]
                        # β 波 14-30Hz
                        beta = bandpass_filter(
                            sig[np.newaxis, :], srate=self.srate,
                            l_freq=14, h_freq=30, order=3)[0]
                        # 去直流 + 归一化
                        alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-8)
                        beta  = (beta  - beta.mean())  / (beta.std()  + 1e-8)
                        _wave_seq += 1
                        tmp_path = wave_path + '.tmp'
                        with open(tmp_path, 'w') as wf:
                            wf.write(f"{_wave_seq}\n")
                            wf.write(f"{','.join(f'{v*50:.1f}' for v in alpha)}\n"
                                     f"{','.join(f'{v*50:.1f}' for v in beta)}")
                        os.replace(tmp_path, wave_path)
                    except: pass

                    if self.model_type == "tcn":
                        import torch
                        x = torch.from_numpy(
                            filtered_norm.astype(np.float32)
                        ).unsqueeze(0)
                        with torch.no_grad():
                            logits = self.model(x)
                            prob = torch.softmax(logits, dim=-1)
                            pred = int(prob.argmax(-1).item())
                            proba = float(prob.max().item())
                    else:
                        features = extract_time_frequency_features(
                            filtered_norm[np.newaxis, :, :], srate=self.srate)
                        features_scaled = self.scaler.transform(features)
                        probs_raw = self.model.predict_proba(features_scaled)[0]
                        # Laplace 平滑: 每类至少 ~4%, 避免 0%
                        probs = (probs_raw + 0.05) / 1.15
                        pred = int(np.argmax(probs))
                        proba = float(probs[pred])
                    self._server.publish_state(pred, proba)
                    self._controller.process_state(pred, proba)
                    self._n += 1
                    if self._n % 50 == 0:
                        if self.model_type != "tcn":
                            p0, p1, p2 = float(probs[0]), float(probs[1]), float(probs[2])
                            logger.info(f"  窗口#{self._n} | {labels.get(pred,'?')}({proba:.0%}) "
                                        f"| 涣散:{p0:.0%} 一般:{p1:.0%} 集中:{p2:.0%} | seq={self._server._seq}")
                        else:
                            logger.info(f"  窗口#{self._n} | 状态: {labels.get(pred,'?')} | 置信度: {proba:.2%} | seq={self._server._seq}")
                time.sleep(self.skip_ms / 1000)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._server: self._server.stop()
        self.device.close()
        logger.info(f"已停止 (共 {self._n} 窗口)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="simulate", choices=["lsl","metabci_lsl","neuracle_tcp","simulate"])
    p.add_argument("--model", default="mlp", choices=["mlp","tcn"],
                   help="分类模型: mlp=手工特征+MLP(67%%), tcn=TCN-Attention端到端")
    p.add_argument("--n-channels", type=int, default=2)
    p.add_argument("--srate", type=int, default=250)
    p.add_argument("--window-size", type=float, default=1.0)
    p.add_argument("--feedback-port", type=int, default=9999)
    p.add_argument("--stream-name", default="")
    p.add_argument("--lsl-timeout", type=float, default=10.0)
    p.add_argument("--decimate", type=int, default=1,
                   help="Neuracle降采样因子 (4=1000→250Hz, 1=不降采样)")
    p.add_argument("--tcp-channels", type=int, default=3,
                   help="Neuracle TCP流通道数 (默认3=Fp1+Fp2+标记)")
    p.add_argument("--adc-gain", type=float, default=1.0,
                   help="ADC→μV转换系数 (Neuracle典型值≈0.02235)")
    args = p.parse_args()

    md = os.path.join(os.path.dirname(__file__), "models")

    if args.model == "tcn":
        # TCN-Attention: 端到端深度学习模型
        import torch
        tcn_path = os.path.join(md, "tcn_attention.pt")
        if not os.path.exists(tcn_path):
            logger.error("TCN 模型未找到: %s (请先运行 train_tcn.py)", tcn_path)
            sys.exit(1)
        clf = torch.jit.load(tcn_path)
        clf.eval()
        scaler = None
        logger.info(">>> 已加载 TCN-Attention 模型 (端到端深度学习) <<<")
    else:
        # MLP: 手工特征 + 标准化 + 分类
        import joblib
        rf = os.path.join(md, "real_classifier.pkl")
        rs = os.path.join(md, "real_scaler.pkl")
        if os.path.exists(rf) and os.path.exists(rs):
            clf = joblib.load(rf)
            scaler = joblib.load(rs)
            logger.info(">>> 已加载真实模型 (MEMA训练, MLP 67.05%%) <<<")
        else:
            clf = joblib.load(os.path.join(md, "attention_classifier.pkl"))
            scaler = joblib.load(os.path.join(md, "scaler.pkl"))
            logger.info(">>> 回退: 加载合成模型 <<<")

    if args.device == "neuracle_tcp":
        device = NeuracleTCPAdapter(n_channels=args.n_channels, srate=args.srate,
                                     decimate=args.decimate,
                                     tcp_channels=args.tcp_channels,
                                     adc_gain=args.adc_gain)
    elif args.device == "metabci_lsl":
        device = MetaBCILSLAdapter(n_channels=args.n_channels, srate=args.srate, stream_name=args.stream_name, timeout=args.lsl_timeout)
    elif args.device == "lsl":
        device = LSLDeviceAdapter(n_channels=args.n_channels, srate=args.srate, stream_name=args.stream_name, timeout=args.lsl_timeout)
    else:
        device = SimulatedDeviceAdapter(n_channels=args.n_channels, srate=args.srate)

    runner = DeviceRunner(
        device=device, model=clf, scaler=scaler,
        window_size=args.window_size, srate=args.srate,
        feedback_port=args.feedback_port, device_type=args.device,
        model_type=args.model,
    )
    runner.start()

if __name__ == "__main__":
    main()
