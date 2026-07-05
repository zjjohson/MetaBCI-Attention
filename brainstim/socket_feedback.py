# -*- coding: utf-8 -*-
"""
Socket 闭环反馈控制组件 (brainstim 新增)

通过标准 Socket 协议对外发布注意力解码状态，联动外部应用实现环境自适应调节。

功能:
    - AttentionFeedbackServer: Socket 服务端，部署在 brainstim 侧
    - AttentionFeedbackClient: Socket 客户端，供外部应用连接
    - 支持实时推送解码状态 + 接收外部控制指令

协议:
    - 元信息:     "META:device:<type>"  (客户端连上时首先发送, type=simulate/neuracle_tcp/lsl)
    - 消息格式:   "STATE:<label>:<confidence>"  (标签:0-2, 置信度:0.0-1.0)
    - 控制指令:   "CMD:<action>"

Authors: 上大Meta梦
Date: 2026-05
License: MIT
"""
import socket
import threading
import json
import time
import logging
from typing import Optional, Callable, Dict

logger = logging.getLogger("attention_feedback")


class AttentionFeedbackServer:
    """
    注意力解码状态 Socket 发布服务器

    作为 brainstim 的扩展组件，实时向外部应用广播注意力状态。

    Parameters
    ----------
    host : str
        绑定地址，默认 0.0.0.0
    port : int
        绑定端口，默认 9999
    max_clients : int
        最大并发客户端数，默认 5
    device_type : str
        设备类型标识: "simulate", "neuracle_tcp", "lsl" 等

    Examples
    --------
    >>> server = AttentionFeedbackServer(port=9999, device_type="neuracle_tcp")
    >>> server.start()  # 后台线程启动
    >>> server.publish_state(0, 0.85)  # 发布: 涣散, 置信度 0.85
    >>> server.publish_state(2, 0.92)  # 发布: 集中, 置信度 0.92
    >>> server.stop()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9999,
        max_clients: int = 5,
        device_type: str = "unknown",
    ):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.device_type = device_type  # "simulate" / "neuracle_tcp" / "lsl"

        self._server_socket: Optional[socket.socket] = None
        self._clients: list = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 回调: 当收到外部控制指令时触发
        self.command_callbacks: Dict[str, Callable] = {}
        self._seq = 0  # STATE 消息序列号

    def start(self) -> None:
        """启动 Socket 服务器（后台线程）"""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(self.max_clients)
        self._server_socket.settimeout(1.0)
        self._running = True

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info(f"[FeedbackServer] 已启动于 {self.host}:{self.port} (设备:{self.device_type})")

    def stop(self) -> None:
        """停止服务器"""
        self._running = False
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

        if self._thread:
            self._thread.join(timeout=2)

        logger.info("[FeedbackServer] 已停止")

    def _accept_loop(self) -> None:
        """接受客户端连接的主循环"""
        while self._running:
            try:
                client, addr = self._server_socket.accept()
                with self._lock:
                    if len(self._clients) < self.max_clients:
                        self._clients.append(client)
                        logger.info(f"[FeedbackServer] 新客户端: {addr}")
                        # === 第一时间发送设备类型元信息 ===
                        try:
                            meta = f"META:device:{self.device_type}\n"
                            client.send(meta.encode("utf-8"))
                        except Exception:
                            pass
                        # 为每个客户端启动接收线程
                        recv_thread = threading.Thread(
                            target=self._recv_loop,
                            args=(client, addr),
                            daemon=True,
                        )
                        recv_thread.start()
                    else:
                        client.close()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"[FeedbackServer] 接受连接错误: {e}")

    def _recv_loop(self, client: socket.socket, addr: tuple) -> None:
        """接收客户端控制指令"""
        client.settimeout(1.0)
        while self._running:
            try:
                data = client.recv(1024)
                if not data:
                    break
                message = data.decode("utf-8").strip()
                self._handle_command(message, addr)
            except socket.timeout:
                continue
            except Exception:
                break

        with self._lock:
            if client in self._clients:
                self._clients.remove(client)
        try:
            client.close()
        except Exception:
            pass

    def _handle_command(self, message: str, addr: tuple) -> None:
        """处理外部控制指令"""
        if not message.startswith("CMD:"):
            return
        action = message[4:].strip()
        logger.info(f"[FeedbackServer] 收到指令 {action} 来自 {addr}")

        if action in self.command_callbacks:
            try:
                self.command_callbacks[action]()
            except Exception as e:
                logger.error(f"[FeedbackServer] 回调执行失败: {e}")

    def publish_state(self, label: int, confidence: float) -> None:
        """
        发布注意力解码状态

        Parameters
        ----------
        label : int
            0=涣散, 1=一般, 2=集中
        confidence : float
            置信度 [0.0, 1.0]
        """
        self._seq += 1
        message = f"STATE:{label}:{confidence:.3f}:{self._seq}\n"
        with self._lock:
            disconnected = []
            for client in self._clients:
                try:
                    client.send(message.encode("utf-8"))
                except Exception:
                    disconnected.append(client)

            for client in disconnected:
                if client in self._clients:
                    self._clients.remove(client)

    def publish_wave(self, fp1, fp2):
        """发布原始滤波波形(各通道降采样到50点)"""
        msg = f"WAVE:{','.join(f'{v:.2f}' for v in fp1[:50])}|{','.join(f'{v:.2f}' for v in fp2[:50])}\n"
        with self._lock:
            dead = []
            for client in self._clients:
                try: client.send(msg.encode())
                except: dead.append(client)
            for c in dead:
                if c in self._clients: self._clients.remove(c)

    def register_command(self, action: str, callback: Callable) -> None:
        """注册指令回调"""
        self.command_callbacks[action] = callback


class AttentionFeedbackClient:
    """
    注意力状态接收客户端

    供外部应用（如桌面便签、白噪音播放器）连接 brainstim Socket 服务器。

    Examples
    --------
    >>> client = AttentionFeedbackClient(port=9999)
    >>> client.connect()
    >>> state = client.get_current_state()  # (2, 0.92) = (集中, 0.92)
    """

    def __init__(self, host: str = "localhost", port: int = 9999):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._current_state: tuple = (-1, 0.0)  # (label, confidence)
        self._current_seq: int = -1  # 最新序列号
        self.device_type: str = ""  # 服务端告知的设备类型
        self._meta_parsed = False

    def connect(self, timeout: float = 5.0) -> bool:
        """连接到服务器"""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        try:
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(0.1)
            self._meta_parsed = False
            return True
        except Exception as e:
            logger.error(f"[FeedbackClient] 连接失败: {e}")
            return False

    def update(self) -> Optional[tuple]:
        """非阻塞地获取最新状态; 同时解析 META 元信息.

        Returns
        -------
        (label, confidence, seq) 或 None
        """
        if not self._socket:
            return None
        try:
            data = self._socket.recv(4096)
            if data:
                for line in data.decode("utf-8").strip().split("\n"):
                    if line.startswith("META:device:"):
                        self.device_type = line.split(":", 2)[2].strip()
                        self._meta_parsed = True
                    elif line.startswith("STATE:"):
                        parts = line.split(":")
                        if len(parts) >= 3:
                            label = int(parts[1])
                            confidence = float(parts[2])
                            seq = int(parts[3]) if len(parts) >= 4 else -1
                            self._current_seq = seq
                            self._current_state = (label, confidence)
                return self._current_state
        except socket.timeout:
            pass
        except Exception:
            pass
        return self._current_state

    def get_current_state(self) -> tuple:
        """获取当前状态: (label, confidence)"""
        return self._current_state

    def send_command(self, action: str) -> bool:
        """发送控制指令"""
        if not self._socket:
            return False
        try:
            self._socket.send(f"CMD:{action}\n".encode("utf-8"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 自适应调控策略
# ---------------------------------------------------------------------------

class AdaptiveController:
    """
    基于注意力状态的自适应调控策略

    根据实时解码状态自动执行:
        - 集中 (2): 屏蔽外部干扰 (关闭通知、降低音量等)
        - 涣散 (0): 白噪音提醒、轻推通知
        - 一般 (1): 维持当前状态

    Examples
    --------
    >>> controller = AdaptiveController()
    >>> controller.register_actions(
    ...     on_focused=lambda: print("屏蔽干扰"),
    ...     on_distracted=lambda: print("播放白噪音"),
    ... )
    >>> controller.process_state(2, 0.85)  # 集中 → 触发 on_focused
    """

    STATE_LABELS = {0: "涣散", 1: "一般", 2: "集中"}

    def __init__(self, min_confidence: float = 0.5, cooldown_seconds: float = 3.0):
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self._last_action_time = 0
        self._last_state = -1

        # 自适应动作
        self._on_focused = None
        self._on_distracted = None
        self._on_normal = None

    def register_actions(
        self,
        on_focused: Optional[Callable] = None,
        on_distracted: Optional[Callable] = None,
        on_normal: Optional[Callable] = None,
    ):
        """注册自适应动作回调"""
        self._on_focused = on_focused
        self._on_distracted = on_distracted
        self._on_normal = on_normal

    def process_state(self, label: int, confidence: float) -> Optional[str]:
        """
        处理注意力状态并触发自适应动作

        Returns
        -------
        触发的动作名称或 None
        """
        if confidence < self.min_confidence:
            return None

        now = time.time()
        # 无论状态是否变化, 冷却期内都不重复触发
        if now - self._last_action_time < self.cooldown_seconds:
            return None

        self._last_state = label
        self._last_action_time = now

        if label == 2 and self._on_focused:
            self._on_focused()
            return "focused"
        elif label == 0 and self._on_distracted:
            self._on_distracted()
            return "distracted"
        elif label == 1 and self._on_normal:
            self._on_normal()
            return "normal"
        return None
