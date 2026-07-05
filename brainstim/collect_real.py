# -*- coding: utf-8 -*-
"""
脑电数据采集刺激界面 (brainstim 框架 + PsychoPy)

通过 Experiment.register_paradigm() 注册到 brainstim 框架。
三阶段: 集中任务(心算) → 一般状态(注视十字) → 涣散休息(闭眼)

用法:
    c:\...\meta\.venv310\Scripts\python.exe c:\...\meta\brainstim\collect_real.py --subject 1 --blocks 3

Authors: 上大Meta梦  |  License: MIT
"""
import sys, os, time, json, argparse, logging

import numpy as _np
for _o, _n in {"alltrue": "all", "sometrue": "any", "product": "prod"}.items():
    if not hasattr(_np, _o): setattr(_np, _o, getattr(_np, _n))
import numpy as np

from psychopy import visual, event, core
from metabci.brainstim.framework import Experiment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("collect")

REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "attention_dataset", "real_data")
os.makedirs(REAL_DATA_DIR, exist_ok=True)

MATH_PROBLEMS = [
    ("43 + 28 = ?", "71"), ("76 - 39 = ?", "37"), ("15 × 6 = ?", "90"),
    ("52 + 37 = ?", "89"), ("84 - 56 = ?", "28"), ("23 × 4 = ?", "92"),
    ("67 + 19 = ?", "86"), ("91 - 45 = ?", "46"), ("58 + 16 = ?", "74"),
]


def run(subject_id=1, n_blocks=3, task_sec=20, normal_sec=12, rest_sec=12,
        screen_id=0, is_fullscr=False):
    """启动 brainstim 框架采集范式"""
    from run_device import NeuracleTCPAdapter

    device = NeuracleTCPAdapter(n_channels=2, srate=250)
    if not device.connect():
        logger.error("Neuracle 连接失败!")
        return

    ex = Experiment(
        monitor=None,
        bg_color_warm=np.array([0.03, 0.04, 0.08]),
        screen_id=screen_id,
        win_size=np.array([1920, 1080] if is_fullscr else [1200, 750]),
        is_fullscr=is_fullscr,
        record_frames=False,
    )

    # ---- paradigm 函数 ----
    def paradigm_loop(win):
        win.setMouseVisible(True)
        W, H = win.size

        # 视觉元素
        main_text = visual.TextStim(win, text="准备开始...", height=42, bold=True,
                                     color="#e2e8f0", units="pix", pos=(0, 80))
        cross_text = visual.TextStim(win, text="+", height=48, color="#64748b",
                                      units="pix", pos=(0, 80))
        hint_text = visual.TextStim(win, text="戴上脑电帽, 准备采集", height=18,
                                     color="#64748b", units="pix", pos=(0, -80))
        status_text = visual.TextStim(win, text="", height=14, color="#3b82f6",
                                       units="pix", pos=(0, -H / 2 + 60))
        timer_text = visual.TextStim(win, text="", height=16, color="#94a3b8",
                                      units="pix", pos=(0, -H / 2 + 90))

        # 状态
        subject_dir = os.path.join(REAL_DATA_DIR, f"sub-{str(subject_id).zfill(2)}")
        os.makedirs(subject_dir, exist_ok=True)

        phases = [
            ("集中任务", 2, task_sec, "#10b981"),
            ("一般状态", 1, normal_sec, "#f59e0b"),
            ("涣散休息", 0, rest_sec, "#ef4444"),
        ]
        hints_map = {
            "集中任务": "请保持专注, 心算答题",
            "一般状态": "注视十字, 放松",
            "涣散休息": "请闭眼休息, 放空大脑",
        }

        _trial = [0]
        _labels = []
        _buf = np.zeros((2, 0))
        _mi = [0]
        _math_q, _math_a = MATH_PROBLEMS[0]
        _last_math_change = -1

        def next_math():
            q, a = MATH_PROBLEMS[_mi[0] % len(MATH_PROBLEMS)]
            _mi[0] += 1
            return q, a

        def beep(freq, dur):
            try:
                import winsound; winsound.Beep(freq, dur)
            except Exception:
                pass

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

        logger.info("采集开始 - subject=%d blocks=%d", subject_id, n_blocks)
        beep(1000, 200)

        # ---- 主循环 ----
        for block in range(n_blocks):
            for phase_name, phase_label, dur, color in phases:
                phase_start = time.time()
                beep(800, 300)
                _last_math_change = -1

                while True:
                    elapsed = time.time() - phase_start
                    remaining = max(0, dur - elapsed)
                    if remaining <= 0:
                        break

                    save_data(phase_label)

                    # 更新画面
                    if phase_name == "集中任务":
                        if int(elapsed) > 0 and int(elapsed) % 8 == 0 and int(elapsed) != _last_math_change:
                            _math_q, _math_a = next_math()
                            _last_math_change = int(elapsed)
                        main_text.text = _math_q
                        main_text.color = color
                        hint_text.text = f"答案: {_math_a} — {hints_map[phase_name]}"
                        main_text.draw()
                    elif phase_name == "一般状态":
                        cross_text.draw()
                        main_text.text = ""
                        hint_text.text = hints_map[phase_name]
                    else:
                        main_text.text = "请闭眼休息"
                        main_text.color = color
                        hint_text.text = hints_map[phase_name]
                        main_text.draw()

                    hint_text.draw()
                    status_text.text = f"Block {block + 1}/{n_blocks} | {phase_name} | 已采集 {_trial[0]} 段"
                    timer_text.text = f"{int(remaining // 60):d}:{int(remaining % 60):02d}"
                    status_text.draw()
                    timer_text.draw()
                    win.flip()

                    keys = event.getKeys(keyList=["escape", "q"])
                    if keys:
                        device.close()
                        logger.info("用户中断采集")
                        return

        # 完成
        device.close()
        meta = {"subject": subject_id, "blocks": n_blocks, "labels": _labels,
                "counts": {str(i): _labels.count(i) for i in [0, 1, 2]}}
        with open(os.path.join(subject_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("完成! 总段数=%d 集中=%d 一般=%d 涣散=%d",
                    _trial[0], _labels.count(2), _labels.count(1), _labels.count(0))

        complete_text = visual.TextStim(win, text="采集完成!", height=36, bold=True,
                                         color="#22c55e", units="pix")
        complete_text.draw()
        win.flip()
        core.wait(2)
        win.close()

    # ---- 注册 & 运行 ----
    ex.register_paradigm("注意力采集", paradigm_loop)

    _orig_close = ex.closeEvent
    ex.closeEvent = lambda: ex.current_win and ex.current_win.close()

    ex.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--task", type=int, default=20)
    p.add_argument("--normal", type=int, default=12)
    p.add_argument("--rest", type=int, default=12)
    p.add_argument("--fullscr", action="store_true")
    args = p.parse_args()
    run(subject_id=args.subject, n_blocks=args.blocks,
        task_sec=args.task, normal_sec=args.normal, rest_sec=args.rest,
        is_fullscr=args.fullscr)
