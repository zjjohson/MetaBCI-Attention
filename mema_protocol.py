# -*- coding: utf-8 -*-
"""

三阶段刺激:
    Neutral       — 1分钟空白屏, 一般状态 (label=1)
    Relaxing      — 5分钟舒缓视频, 涣散休息 (label=0)
    Concentrating — 5分钟课程视频, 集中学习 (label=2)

每阶段后: 自我评估 (注意力自评 + VAD情绪量表)

用法:
    python mema_protocol.py --subject 1 --blocks 1
"""
import sys, os, time, json, argparse, logging, threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("mema")

REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "attention_dataset", "real_data")
os.makedirs(REAL_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 媒体文件路径 (把 mp4/mp3 放在 media/ 目录下, 或改为你的实际路径)
# Neutral: 放空白/黑屏视频, 或设为 None 只用 tkinter 黑色背景
# Relaxing: 风景舒缓视频 + 放松音乐
# Concentrating: 课程/讲座视频
# ---------------------------------------------------------------------------
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

TASKS = [
    # (name, label, duration_sec, color, instruction, video_file, audio_file)
    ("Neutral",      1, 60,   "#f59e0b", "保持正常状态, 注视屏幕",
        None, None),                                    # ← 放空白视频或 None
    ("Relaxing",     0, 300,  "#ef4444", "请尽量放松, 减少专注",
        "relaxing_video.mp4", "relaxing_music.mp3"),    # ← 替换为你的文件
    ("Concentrating", 2, 300, "#10b981", "请集中注意力, 主动学习思考",
        "lecture_video.mp4", None),                     # ← 替换为你的文件
]


def play_media(video_file, audio_file):
    """在 tkinter 窗口右侧用 subprocess 打开视频, pygame 播放背景音乐"""
    import subprocess
    # ==== 音频 (pygame mixer) ====
    if audio_file:
        audio_path = os.path.join(MEDIA_DIR, audio_file)
        if os.path.exists(audio_path):
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play(-1)  # 循环播放
                logger.info(f"♪ 播放音乐: {audio_file}")
            except Exception as e:
                logger.warning(f"音频失败: {e}")

    # ==== 视频 (用系统默认播放器在新窗口打开) ====
    if video_file:
        video_path = os.path.join(MEDIA_DIR, video_file)
        if os.path.exists(video_path):
            try:
                # Windows: 用默认播放器打开视频
                os.startfile(video_path)
                logger.info(f"▶ 播放视频: {video_file}")
            except Exception as e:
                logger.warning(f"视频失败: {e}")


def stop_media():
    """停止所有媒体"""
    try:
        import pygame
        pygame.mixer.music.stop()
    except:
        pass


def run(subject_id=1, n_blocks=1):
    import tkinter as tk
    from tkinter import ttk
    from run_device import NeuracleTCPAdapter
    from attention_dataset.dataset import bandpass_filter

    device = NeuracleTCPAdapter(n_channels=2, srate=250)
    if not device.connect():
        logger.error("Neuracle 连接失败!")
        return

    subject_dir = os.path.join(REAL_DATA_DIR, f"sub-{str(subject_id).zfill(2)}")
    os.makedirs(subject_dir, exist_ok=True)

    # ---- 主窗口 ----
    root = tk.Tk()
    root.title("MEMA Protocol — 注意力脑电采集")
    root.configure(bg="#080c18")
    root.geometry("1200x750+0+0")

    # 主提示
    main_lbl = tk.Label(root, text="MEMA 注意力脑电采集协议",
                        font=("Microsoft YaHei", 36, "bold"),
                        fg="#e2e8f0", bg="#080c18")
    main_lbl.pack(pady=(180, 10))

    # 阶段名称
    phase_lbl = tk.Label(root, text="准备开始...",
                         font=("Microsoft YaHei", 24, "bold"),
                         fg="#3b82f6", bg="#080c18")
    phase_lbl.pack(pady=(20, 0))

    # 指令
    instr_lbl = tk.Label(root, text="戴上脑电帽, 准备采集",
                         font=("Microsoft YaHei", 16),
                         fg="#64748b", bg="#080c18")
    instr_lbl.pack(pady=(20, 0))

    # 进度条
    progress = ttk.Progressbar(root, length=800, mode='determinate')
    progress.pack(pady=(40, 0))

    # 计时器
    timer_lbl = tk.Label(root, text="", font=("Consolas", 14),
                         fg="#94a3b8", bg="#080c18")
    timer_lbl.pack(pady=(10, 0))

    # 状态栏
    status_lbl = tk.Label(root, text="", font=("Microsoft YaHei", 12),
                          fg="#3b82f6", bg="#080c18")
    status_lbl.pack(side="bottom", pady=(0, 30))

    # ---- 采集状态 ----
    _trial = [0]
    _labels = []
    _buf = np.zeros((2, 0))
    _block = [0]
    _phase = [0]
    _phase_start = [0.0]
    _paused = [True]  # 初始暂停, 空格开始

    def save_data():
        nonlocal _buf
        data = device.get_data(n_samples=50)
        if data is not None and data.shape[1] >= 2:
            _buf = np.concatenate([_buf, data.astype(np.float64)], axis=-1)
        while _buf.shape[1] >= 1000:
            seg = _buf[:, :1000:4]  # 1000Hz → 250Hz
            np.save(os.path.join(subject_dir, f"trial_{_trial[0]:04d}.npy"), seg)
            _labels.append(TASKS[_phase[0]][1])
            _trial[0] += 1
            _buf = _buf[:, 500:]

    def self_assessment():
        """注意力 + VAD 自评弹窗"""
        dlg = tk.Toplevel(root)
        dlg.title("自评")
        dlg.configure(bg="#131a2a")
        dlg.geometry("500x450+350+150")
        dlg.grab_set()

        results = {}

        tk.Label(dlg, text="注意力自评", font=("Microsoft YaHei", 18, "bold"),
                 fg="#e2e8f0", bg="#131a2a").pack(pady=(20, 10))

        attn_var = tk.StringVar(value="一般")
        for txt, val in [("集中 (Concentrating)", "集中"),
                         ("一般 (Neutral)", "一般"),
                         ("涣散 (Relaxing)", "涣散")]:
            tk.Radiobutton(dlg, text=txt, variable=attn_var, value=val,
                           font=("Microsoft YaHei", 14),
                           fg="#e2e8f0", bg="#131a2a",
                           selectcolor="#1a2636").pack(anchor="w", padx=60, pady=4)
        results["attention"] = attn_var

        tk.Label(dlg, text="VAD 情绪量表 (1-5分)",
                 font=("Microsoft YaHei", 18, "bold"),
                 fg="#e2e8f0", bg="#131a2a").pack(pady=(20, 10))

        for dim, desc in [("Valence", "愉悦度 (1=很不愉快, 5=很愉快)"),
                          ("Arousal", "唤醒度 (1=很平静, 5=很兴奋)"),
                          ("Dominance", "支配度 (1=很被动, 5=很主动)")]:
            frm = tk.Frame(dlg, bg="#131a2a")
            frm.pack(anchor="w", padx=60, pady=4)
            tk.Label(frm, text=f"{dim}:", font=("Microsoft YaHei", 13),
                     fg="#94a3b8", bg="#131a2a").pack(side="left")
            var = tk.IntVar(value=3)
            results[dim] = var
            for v in range(1, 6):
                tk.Radiobutton(frm, text=str(v), variable=var, value=v,
                               font=("Microsoft YaHei", 11),
                               fg="#e2e8f0", bg="#131a2a",
                               selectcolor="#1a2636").pack(side="left", padx=2)

        def submit():
            dlg.destroy()

        tk.Button(dlg, text="提交", command=submit,
                  font=("Microsoft YaHei", 14), bg="#3b82f6", fg="white",
                  relief="flat", padx=30, pady=8).pack(pady=(20, 10))
        dlg.wait_window()

        # 保存自评
        r = {"attention": results["attention"].get(),
             "valence": results["Valence"].get(),
             "arousal": results["Arousal"].get(),
             "dominance": results["Dominance"].get(),
             "phase": TASKS[_phase[0]][0]}
        with open(os.path.join(subject_dir, "self_assess.jsonl"), "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"自评: {r}")

    def tick():
        nonlocal _buf
        if _block[0] >= n_blocks:
            device.close()
            meta = {"subject": subject_id, "blocks": n_blocks, "labels": _labels,
                    "counts": {str(i): _labels.count(i) for i in [0, 1, 2]},
                    "protocol": "MEMA"}
            with open(os.path.join(subject_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            logger.info("完成! 总段数=%d 集中=%d 一般=%d 涣散=%d",
                        _trial[0], _labels.count(2), _labels.count(1), _labels.count(0))
            root.destroy()
            return

        if _paused[0]:
            status_lbl.config(text="按 空格键 开始当前阶段")
            root.after(200, tick)
            return

        name, label, dur, color, instr, video, audio = TASKS[_phase[0]]
        elapsed = time.time() - _phase_start[0]
        remaining = max(0, dur - elapsed)

        save_data()

        # 更新 UI
        phase_lbl.config(text=name, fg=color)
        instr_lbl.config(text=instr)
        m, s = int(remaining // 60), int(remaining % 60)
        timer_lbl.config(text=f"{m:02d}:{s:02d}")
        progress['value'] = (1 - remaining / dur) * 100
        status_lbl.config(
            text=f"Block {_block[0]+1}/{n_blocks} | {name} | 已采集 {_trial[0]} 段")
        root.update()

        if remaining <= 0:
            stop_media()  # ★ 停止视频和音乐
            # 自评
            progress['value'] = 100
            self_assessment()

            # 下一阶段
            _phase[0] += 1
            if _phase[0] >= len(TASKS):
                _phase[0] = 0
                _block[0] += 1
                if _block[0] >= n_blocks:
                    root.after(200, tick)
                    return
            _phase_start[0] = time.time()
            _buf = np.zeros((2, 0))  # 清缓冲, 新阶段
            _paused[0] = True

        root.after(200, tick)

    # 空格键控制
    def space_key(e):
        if _paused[0]:
            _paused[0] = False
            _phase_start[0] = time.time()
            _, _, _, _, _, video, audio = TASKS[_phase[0]]
            play_media(video, audio)   # ★ 播放当前阶段视频+音乐
            status_lbl.config(text="采集中...")
            logger.info("▶ 开始: %s", TASKS[_phase[0]][0])

    root.bind("<space>", space_key)
    root.after(500, tick)
    root.mainloop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--blocks", type=int, default=1,
                   help="重复次数 (1 block = Neutral+Relaxing+Concentrating)")
    args = p.parse_args()
    run(subject_id=args.subject, n_blocks=args.blocks)
