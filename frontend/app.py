# -*- coding: utf-8 -*-
"""
MetaBCI 注意力监测仪表盘 — 精简重写版
- 演示模式 / 设备模式 互斥，用户按钮显式切换
- 启动时一次性检测设备，不自动抢夺
- threading.Event 控制线程启停，不残留
"""
import sys, os, time, json, random, threading, queue, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, render_template, Response, jsonify, request

# ---------------------------------------------------------------------------
# Flask 初始化
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

@app.after_request
def no_cache(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = r.headers["Expires"] = "no-cache"
    return r

# ---------------------------------------------------------------------------
# 状态管理器 (保持不变 — 已验证稳定)
# ---------------------------------------------------------------------------
class StateManager:
    def __init__(self):
        self._lk = threading.Lock()
        self.state = {"label": -1, "confidence": 0.0, "timestamp": ""}
        self.history = []
        self._sse = []

    def update(self, label, confidence, timestamp=""):
        from datetime import datetime
        ts = timestamp or datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with self._lk:
            self.state = {"label": label, "confidence": confidence, "timestamp": ts}
            self.history.append(dict(self.state))
            if len(self.history) > 200:
                self.history.pop(0)
        data = json.dumps(self.state)
        for q in list(self._sse):
            try:
                q.put_nowait(data)
            except Exception:
                self._sse.remove(q)

    def stats(self):
        with self._lk:
            if not self.history:
                return {"focused": 0, "normal": 0, "distracted": 0, "total": 0, "avg_confidence": 0.0}
            labels = [s["label"] for s in self.history]
            total = len(labels)
            avg = sum(s["confidence"] for s in self.history) / total
            return {"focused": labels.count(2), "normal": labels.count(1),
                    "distracted": labels.count(0), "total": total, "avg_confidence": round(avg, 3)}

    def reset(self):
        with self._lk:
            self.history.clear()
            self.state = {"label": -1, "confidence": 0.0, "timestamp": ""}

    def add_sse(self, q):
        self._sse.append(q)

    def remove_sse(self, q):
        if q in self._sse:
            self._sse.remove(q)


sm = StateManager()

# ---------------------------------------------------------------------------
# 模式管理 (精简 — 全局唯一)
# ---------------------------------------------------------------------------
_mode = "idle"          # "demo" | "device" | "idle"
_mode_thread = None
_mode_stop = threading.Event()
_last_data_time = 0.0   # 设备模式下最后一次收到数据的时间

# 演示数据
FAKE = [2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 1, 0, 2, 2, 2]
_fi = 0


def _demo_loop():
    """演示模式：每 3 秒生成预测"""
    global _fi, _mode
    stop = _mode_stop  # ★ 捕获当前 Event, 不受后续 switch_mode 替换影响
    app.logger.info("▶ 演示模式 启动")
    while not stop.is_set():
        _fi = (_fi + 1) % len(FAKE)
        lbl, conf = FAKE[_fi], round(0.7 + random.random() * 0.3, 3)
        sm.update(lbl, conf)
        stop.wait(3.0)  # 可中断的 sleep
    app.logger.info("■ 演示模式 停止")


def _dev_loop():
    """设备模式：连接 Socket 9999 读取真实数据"""
    global _mode, _last_data_time
    from brainstim.socket_feedback import AttentionFeedbackClient
    stop = _mode_stop  # ★ 捕获当前 Event

    app.logger.info("▶ 设备模式 启动 (连接 9999 ...)")
    c = AttentionFeedbackClient(port=9999)

    if not c.connect():
        app.logger.warning("✗ 无法连接 Socket 9999")
        sm.update(-1, 0.0)
        _mode = "idle"
        return

    # 读取 META
    meta_type = ""
    for _ in range(30):
        c.update()
        if c.device_type:
            meta_type = c.device_type
            break
        if stop.is_set():
            c.close()
            return
        stop.wait(0.1)

    app.logger.info("✓ Socket 已连接 (META=%s)", meta_type or "无")

    _last_data_time = time.time()
    data_count = 0

    while not stop.is_set():
        try:
            s = c.update()
            if s and s[0] >= 0:
                data_count += 1
                _last_data_time = time.time()
                sm.update(s[0], s[1])
                if data_count <= 3:
                    app.logger.info("  dev→update(%d, %.3f) #%d", s[0], s[1], data_count)
        except Exception as e:
            app.logger.error("dev_loop 异常: %s", e)
        stop.wait(1.5)  # 1.5秒刷新一次, 避免前端更新过快

    try:
        c.close()
    except Exception:
        pass
    app.logger.info("■ 设备模式 停止")


def switch_mode(target: str):
    """切换到指定模式：先停当前，再启动新。target ∈ {demo, device, idle}"""
    global _mode, _mode_thread, _mode_stop

    if _mode == target:
        app.logger.info("  已在 %s 模式，不切换", target)
        return

    # 1) 停止当前模式
    if _mode_thread and _mode_thread.is_alive():
        app.logger.info("  停止当前模式: %s", _mode)
        _mode_stop.set()                         # 发停止信号
        _mode_thread.join(timeout=5)             # 等最多 5 秒
        if _mode_thread.is_alive():
            app.logger.warning("  旧线程未能在 5s 内退出")
        # 创建全新 Event 给下一个线程, 避免竞态
        _mode_stop = threading.Event()

    # 2) 启动新模式
    _mode = target
    if target == "demo":
        _mode_thread = threading.Thread(target=_demo_loop, daemon=True, name="demo-loop")
        _mode_thread.start()
    elif target == "device":
        _mode_thread = threading.Thread(target=_dev_loop, daemon=True, name="dev-loop")
        _mode_thread.start()
    else:
        _mode_thread = None


# ---------------------------------------------------------------------------
# 后台设备检测 (仅上报可用性，不自动切换)
# ---------------------------------------------------------------------------
_device_available = False

def _detect_device():
    """后台线程：每 5 秒检测一次 9999 端口是否有设备"""
    global _device_available
    from brainstim.socket_feedback import AttentionFeedbackClient
    while True:
        time.sleep(5)
        try:
            c = AttentionFeedbackClient(port=9999)
            if not c.connect():
                _device_available = False
                continue
            t, has_data = "", False
            for _ in range(15):
                r = c.update()
                if c.device_type:
                    t = c.device_type
                if r and r[0] >= 0:
                    has_data = True
                time.sleep(0.1)
            c.close()
            # 有 META 且非 simulate，或有数据流 → 设备可用
            _device_available = (t and t != "simulate") or (has_data and t != "simulate")
        except Exception:
            _device_available = False


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    return jsonify(sm.state)


@app.route("/api/history")
def api_history():
    n = request.args.get("n", 100, type=int)
    with sm._lk:
        return jsonify(list(sm.history[-min(n, len(sm.history)):]))


@app.route("/api/wave")
def api_wave():
    try:
        wf = os.path.join(os.path.dirname(__file__), "..", ".eeg_wave.txt")
        if os.path.exists(wf):
            with open(wf) as f:
                lines = f.read().strip().split("\n")
                if len(lines) >= 3:
                    return jsonify({
                        "seq": int(float(lines[0].strip())),
                        "fp1": [float(x) for x in lines[1].split(",")],
                        "fp2": [float(x) for x in lines[2].split(",")],
                    })
    except Exception:
        pass
    return jsonify({"seq": 0, "fp1": [], "fp2": []})


@app.route("/api/stats")
def api_stats():
    return jsonify(sm.stats())


@app.route("/stream")
def stream():
    def gen():
        q = queue.Queue(maxsize=100)
        sm.add_sse(q)
        app.logger.info("[SSE] 客户端连接 (队列数=%d)", len(sm._sse))
        try:
            while True:
                try:
                    item = q.get(timeout=5)
                    yield f"data: {item}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            sm.remove_sse(q)
            app.logger.info("[SSE] 客户端断开 (队列数=%d)", len(sm._sse))

    resp = Response(gen(), mimetype="text/event-stream")
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


# ---- 模式切换 (显式，非 toggle) ----

@app.route("/api/mode/demo", methods=["POST"])
def api_mode_demo():
    """切换到演示模式"""
    try:
        app.logger.info("← 用户请求: 演示模式 (当前=%s)", _mode)
        switch_mode("demo")
        sm.reset()
        return jsonify({"mode": _mode, "device_available": _device_available})
    except Exception as e:
        app.logger.exception("api_mode_demo 崩溃")
        return jsonify({"mode": _mode, "error": str(e)}), 500


@app.route("/api/mode/device", methods=["POST"])
def api_mode_device():
    """切换到设备模式"""
    try:
        app.logger.info("← 用户请求: 设备模式 (当前=%s)", _mode)
        switch_mode("device")
        sm.reset()
        return jsonify({"mode": _mode, "device_available": _device_available})
    except Exception as e:
        app.logger.exception("api_mode_device 崩溃")
        return jsonify({"mode": _mode, "error": str(e)}), 500


@app.route("/api/status")
def api_status():
    """当前系统状态"""
    return jsonify({
        "mode": _mode,
        "device_available": _device_available,
        "last_data_sec": round(time.time() - _last_data_time, 1) if _last_data_time else -1,
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    sm.reset()
    return jsonify({"message": "ok"})


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    # 一次性启动检测
    from brainstim.socket_feedback import AttentionFeedbackClient
    qc = AttentionFeedbackClient(port=9999)
    start_mode = "demo"
    if qc.connect():
        t = ""
        for _ in range(10):
            qc.update()
            if qc.device_type:
                t = qc.device_type
                break
            time.sleep(0.1)
        if t and t != "simulate":
            start_mode = "device"
            app.logger.info(">>> 启动检测: 发现设备 (%s), 直接进入设备模式 <<<", t)
        qc.close()

    # 启动背景线程
    threading.Thread(target=_detect_device, daemon=True).start()

    # 进入初始模式
    switch_mode(start_mode)

    app.logger.info("仪表盘: http://localhost:%s  |  初始模式: %s", args.port, start_mode)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
