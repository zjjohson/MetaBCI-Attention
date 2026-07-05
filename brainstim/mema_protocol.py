# -*- coding: utf-8 -*-
"""
MEMA 论文协议刺激界面 (brainstim 框架 + PsychoPy)

三阶段刺激 (MEMA论文协议):
    Neutral       — 1分钟黑色全屏 + 黄色十字, 一般状态 (label=1)
    Concentrating — 5分钟学习视频循环播放, 集中学习 (label=2)
    Relaxing      — 5分钟风景视频循环播放, 涣散休息 (label=0)

每阶段后: 自我评估 (注意力自评 + VAD情绪量表, 键盘数字输入)

用法:
    .venv310\Scripts\python.exe brainstim\mema_protocol.py --subject 1 --blocks 1

Authors: 上大Meta梦  |  License: MIT
"""
import sys, os, time, json, argparse, logging

import numpy as _np
for _o, _n in {"alltrue": "all", "sometrue": "any", "product": "prod"}.items():
    if not hasattr(_np, _o): setattr(_np, _o, getattr(_np, _n))
import numpy as np

from psychopy import visual, event, core
from psychopy.visual import MovieStim
from metabci.brainstim.framework import Experiment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("mema")

REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "attention_dataset", "real_data")
os.makedirs(REAL_DATA_DIR, exist_ok=True)

MEDIA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

def start_audio(audio_file):
    """pygame 播放背景音乐 (循环)"""
    if not audio_file:
        return
    audio_path = os.path.join(MEDIA_DIR, audio_file)
    if os.path.exists(audio_path):
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play(-1)
            logger.info(f"♪ 播放音乐: {audio_file}")
        except Exception as e:
            logger.warning(f"音频失败: {e}")


def start_video(win, video_file):
    """在 PsychoPy 窗口内嵌播放视频, 返回 MovieStim 对象"""
    if not video_file:
        return None
    video_path = os.path.join(MEDIA_DIR, video_file)
    if not os.path.exists(video_path):
        logger.warning(f"视频不存在: {video_file}")
        return None
    try:
        W, H = win.size
        movie = MovieStim(win, video_path, size=(W, H), units="pix", loop=True)
        logger.info(f"▶ 内嵌播放: {video_file}")
        return movie
    except Exception as e:
        logger.warning(f"内嵌视频失败: {e}, 回退外部播放")
        try:
            os.startfile(video_path)
        except Exception:
            pass
        return None


def stop_media(movie=None):
    """停止视频和音乐"""
    if movie is not None:
        try:
            movie.stop()
        except Exception:
            pass
    try:
        import pygame
        pygame.mixer.music.stop()
    except Exception:
        pass


def self_assessment(win, phase_name, subject_dir):
    """PsychoPy 内键盘自评表单 (暗色玻璃卡片)"""
    W, H = win.size

    # 暗色毛玻璃卡片
    bg_w, bg_h = 600, 420
    bg_rect = visual.Rect(win, width=bg_w, height=bg_h, pos=(0, 0),
                           fillColor="#0f172a", opacity=0.92,
                           lineColor="#334155", lineWidth=1, units="pix")

    title = visual.TextStim(win, text="", height=26, bold=True,
                             color="#f1f5f9", units="pix", pos=(0, bg_h / 2 - 70))
    prompt = visual.TextStim(win, text="", height=18, color="#94a3b8",
                              units="pix", pos=(0, 5))
    status = visual.TextStim(win, text="", height=13, color="#60a5fa",
                              units="pix", pos=(0, -bg_h / 2 + 45))
    result = {}

    # 注意力自评
    title.text = "注意力自评"
    prompt.text = ("1 = 涣散 (Relaxing)\n"
                   "2 = 一般 (Neutral)\n"
                   "3 = 集中 (Concentrating)\n\n"
                   "请按 1 / 2 / 3 选择")
    bg_rect.draw(); title.draw(); prompt.draw(); status.draw()
    win.flip()
    keys = event.waitKeys(keyList=["1", "2", "3", "escape"])
    if keys and keys[0] == "escape":
        return None
    attention_map = {"1": "涣散", "2": "一般", "3": "集中"}
    result["attention"] = attention_map.get(keys[0], "一般")
    status.text = f"已选择: {result['attention']}"
    bg_rect.draw(); title.draw(); prompt.draw(); status.draw()
    win.flip()
    core.wait(0.5)

    # VAD 量表
    vad_dims = [
        ("Valence",   "愉悦度\n\n1=很不愉快  2=较不愉快  3=一般\n4=较愉快    5=很愉快"),
        ("Arousal",   "唤醒度\n\n1=很平静    2=较平静    3=一般\n4=较兴奋    5=很兴奋"),
        ("Dominance", "支配度\n\n1=很被动    2=较被动    3=一般\n4=较主动    5=很主动"),
    ]
    for dim, desc in vad_dims:
        title.text = f"VAD 情绪量表 — {dim}"
        prompt.text = desc + "\n\n请按 1-5 选择"
        bg_rect.draw(); title.draw(); prompt.draw(); status.draw()
        win.flip()
        keys = event.waitKeys(keyList=["1", "2", "3", "4", "5", "escape"])
        if keys and keys[0] == "escape":
            return None
        result[dim.lower()] = int(keys[0])
        status.text = f"{dim}: {keys[0]} / 5"
        bg_rect.draw(); title.draw(); prompt.draw(); status.draw()
        win.flip()
        core.wait(0.3)

    record = {
        "attention": result["attention"],
        "valence": result["valence"],
        "arousal": result["arousal"],
        "dominance": result["dominance"],
        "phase": phase_name,
    }
    with open(os.path.join(subject_dir, "self_assess.jsonl"), "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"自评: {record}")
    return record


def run(subject_id=1, n_blocks=1, screen_id=0, is_fullscr=False):
    """启动 brainstim 框架 MEMA 协议范式"""
    from run_device import NeuracleTCPAdapter

    TASKS = [
        ("Neutral",       1, 60,   None, None),
        ("Concentrating", 2, 300,  "lecture_video.mp4", None),
        ("Relaxing",      0, 300,  "relaxing_video.mp4", "relaxing_music.mp3"),
    ]

    device = NeuracleTCPAdapter(n_channels=2, srate=250, tcp_channels=3)
    if not device.connect():
        logger.error("Neuracle 连接失败!")
        return

    ex = Experiment(
        monitor=None,
        bg_color_warm=np.array([0.0, 0.0, 0.0]),  # 纯黑
        screen_id=screen_id,
        win_size=np.array([1920, 1080] if is_fullscr else [1200, 750]),
        is_fullscr=is_fullscr,
        record_frames=False,
    )

    def paradigm_loop(win):
        win.setMouseVisible(False)
        W, H = win.size

        # 仅有的视觉元素：大十字 和 等待提示
        cross = visual.TextStim(win, text="+", height=120, bold=True,
                                 color="#eab308", units="pix", pos=(0, 0))
        wait_text = visual.TextStim(win, text="", height=18, color="#94a3b8",
                                     units="pix", pos=(0, -H / 2 + 45))
        # 等待提示的暗色底托
        wait_bg = visual.Rect(win, width=360, height=32, pos=(0, -H / 2 + 45),
                               fillColor="#0f172a", opacity=0.85,
                               lineColor="#334155", lineWidth=1, units="pix")

        # 数据状态
        subject_dir = os.path.join(REAL_DATA_DIR, f"sub-{str(subject_id).zfill(2)}")
        os.makedirs(subject_dir, exist_ok=True)

        _trial = [0]
        _labels = []
        _buf = np.zeros((2, 0))

        def save_data(label):
            nonlocal _buf
            data = device.get_data(n_samples=50)
            if data is not None and data.shape[1] >= 2:
                _buf = np.concatenate([_buf, data.astype(np.float64)], axis=-1)
            while _buf.shape[1] >= 1000:
                seg = _buf[:, :1000:4]
                np.save(os.path.join(subject_dir, f"trial_{_trial[0]:04d}.npy"), seg)
                _labels.append(label)
                _trial[0] += 1
                _buf = _buf[:, 500:]

        logger.info("MEMA 协议启动 - subject=%d blocks=%d", subject_id, n_blocks)

        # ---- 主循环 ----
        for block in range(n_blocks):
            _phase = [0]
            movie = None

            while _phase[0] < len(TASKS):
                name, label, dur, video, audio = TASKS[_phase[0]]

                # ---- 等待空格 ----
                wait_text.text = f"按 空格键 开始 — {name}"
                wait_bg.draw(); wait_text.draw()
                win.flip()

                keys = event.waitKeys(keyList=["space", "escape", "q"])
                if keys and keys[0] in ("escape", "q"):
                    stop_media(movie)
                    device.close()
                    logger.info("用户退出")
                    return

                # ---- 阶段开始 ----
                phase_start = time.time()
                if name == "Neutral":
                    # 一般: 纯黑 + 黄色十字, 无视频无音频
                    pass
                else:
                    start_audio(audio)
                    movie = start_video(win, video)
                logger.info("▶ 开始: %s (%.0fs)", name, dur)

                # ---- 阶段主循环 ----
                while True:
                    elapsed = time.time() - phase_start
                    remaining = dur - elapsed
                    if remaining <= 0:
                        break

                    save_data(label)

                    if name == "Neutral":
                        # 纯黑背景 + 黄色十字, 无任何文字
                        cross.draw()
                    else:
                        # 视频全屏, 无任何文字覆盖
                        if movie is not None:
                            movie.draw()

                    win.flip()

                    keys = event.getKeys(keyList=["escape", "q"])
                    if keys:
                        stop_media(movie)
                        device.close()
                        logger.info("用户中断")
                        return

                # ---- 阶段结束 ----
                if name != "Neutral":
                    stop_media(movie)
                    movie = None

                self_assessment(win, name, subject_dir)

                _phase[0] += 1
                _buf = np.zeros((2, 0))

        # ---- 完成 ----
        device.close()
        meta = {"subject": subject_id, "blocks": n_blocks, "labels": _labels,
                "counts": {str(i): _labels.count(i) for i in [0, 1, 2]},
                "protocol": "MEMA"}
        with open(os.path.join(subject_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("完成! 总段数=%d 集中=%d 一般=%d 涣散=%d",
                    _trial[0], _labels.count(2), _labels.count(1), _labels.count(0))

        complete_text = visual.TextStim(win, text="采集完成! 按任意键退出", height=30,
                                         bold=True, color="#f1f5f9", units="pix")
        complete_bg = visual.Rect(win, width=460, height=60, pos=(0, 0),
                                   fillColor="#0f172a", opacity=0.92,
                                   lineColor="#334155", lineWidth=1, units="pix")
        complete_bg.draw(); complete_text.draw()
        win.flip()
        event.waitKeys()  # 按任意键返回菜单

    ex.register_paradigm("开始采集", paradigm_loop)

    _orig_close = ex.closeEvent
    ex.closeEvent = lambda: ex.current_win and ex.current_win.close()

    ex.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--blocks", type=int, default=1,
                   help="重复次数 (1 block = Neutral+Concentrating+Relaxing)")
    p.add_argument("--fullscr", action="store_true")
    args = p.parse_args()
    run(subject_id=args.subject, n_blocks=args.blocks, is_fullscr=args.fullscr)
