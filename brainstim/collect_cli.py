# -*- coding: utf-8 -*-
"""
命令行脑电数据采集 (brainstim 框架 + PsychoPy 窗口)

保留终端 print() + 新增 PsychoPy 视觉反馈，双通道显示。

用法:
    c:\...\meta\.venv310\Scripts\python.exe c:\...\meta\brainstim\collect_cli.py --subject 1 --blocks 3

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
logger = logging.getLogger("collect_cli")

REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "attention_dataset", "real_data")
os.makedirs(REAL_DATA_DIR, exist_ok=True)


def beep(freq=800, duration=300):
    try:
        import winsound; winsound.Beep(freq, duration)
    except Exception:
        print("\a")


def run(subject_id=1, n_blocks=3, task_sec=20, normal_sec=12, rest_sec=12,
        screen_id=0, is_fullscr=False, srate=250):
    """启动 brainstim 框架 CLI 采集范式"""
    from run_device import NeuracleTCPAdapter
    from attention_dataset.dataset import bandpass_filter

    device = NeuracleTCPAdapter(n_channels=2, srate=srate)
    if not device.connect():
        logger.error("Neuracle 连接失败 (127.0.0.1:8712)")
        return

    ex = Experiment(
        monitor=None,
        bg_color_warm=np.array([0.03, 0.04, 0.08]),
        screen_id=screen_id,
        win_size=np.array([1920, 1080] if is_fullscr else [800, 600]),
        is_fullscr=is_fullscr,
        record_frames=False,
    )

    def paradigm_loop(win):
        win.setMouseVisible(True)
        W, H = win.size

        phase_text = visual.TextStim(win, text="", height=28, bold=True,
                                      color="#e2e8f0", units="pix", pos=(0, H / 4))
        hint_text = visual.TextStim(win, text="", height=20, color="#64748b",
                                     units="pix", pos=(0, 0))
        timer_text = visual.TextStim(win, text="", height=24, color="#94a3b8",
                                      units="pix", pos=(0, -H / 4))
        status_text = visual.TextStim(win, text="", height=14, color="#3b82f6",
                                       units="pix", pos=(0, -H / 2 + 40))

        subject_dir = os.path.join(REAL_DATA_DIR, f"sub-{str(subject_id).zfill(2)}")
        os.makedirs(subject_dir, exist_ok=True)

        phases = [
            ("集中任务", 2, "请做心算 (如 43+28=?) 保持专注"),
            ("一般状态", 1, "注视屏幕中央, 放松"),
            ("涣散休息", 0, "请闭眼休息, 放空大脑"),
        ]

        trial_idx = 0
        all_labels = []
        _buf = np.zeros((2, 0))

        logger.info("CLI 采集开始 - subject=%d blocks=%d", subject_id, n_blocks)

        for block in range(n_blocks):
            for phase_name, phase_label, hint in phases:
                dur = task_sec if phase_label == 2 else (normal_sec if phase_label == 1 else rest_sec)

                print(f"\n{'='*50}")
                print(f"  Block {block+1}/{n_blocks} | {phase_name}")
                print(f"  {hint}")
                print(f"  时长: {dur} 秒")
                print(f"{'='*50}")

                beep(1000, 200)
                time.sleep(0.3)
                beep(1200, 200)

                t_start = time.time()
                while time.time() - t_start < dur:
                    remaining = dur - (time.time() - t_start)
                    print(f"\r  剩余 {remaining:.0f}s | {phase_name} | 已采集 {trial_idx} 段", end="", flush=True)

                    data = device.get_data(n_samples=50)
                    if data is not None:
                        _buf = np.concatenate([_buf, data.astype(np.float64)], axis=-1)
                        # 累积到 1000 点 → 降采样 → 250 点 = 1.0s
                        while _buf.shape[1] >= 1000:
                            seg = _buf[:, :1000:4]
                            np.save(os.path.join(subject_dir, f"trial_{trial_idx:04d}.npy"), seg)
                            all_labels.append(phase_label)
                            trial_idx += 1
                            _buf = _buf[:, 500:]
                    else:
                        time.sleep(0.02)

                    # 更新 PsychoPy 窗口
                    phase_text.text = f"Block {block+1}/{n_blocks} | {phase_name}"
                    hint_text.text = hint
                    timer_text.text = f"剩余 {int(remaining):d}s"
                    status_text.text = f"已采集 {trial_idx} 段"

                    phase_text.draw()
                    hint_text.draw()
                    timer_text.draw()
                    status_text.draw()
                    win.flip()

                    keys = event.getKeys(keyList=["escape", "q"])
                    if keys:
                        device.close()
                        logger.info("用户中断采集")
                        print()
                        return

                beep(600, 400)
                print()

        device.close()

        meta = {
            "subject": subject_id, "blocks": n_blocks,
            "labels": all_labels,
            "counts": {str(i): all_labels.count(i) for i in [0, 1, 2]},
        }
        with open(os.path.join(subject_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n{'='*50}")
        print(f"  完成! 总段数: {trial_idx}")
        for i, name in [(2, "集中"), (1, "一般"), (0, "涣散")]:
            print(f"  {name}({i}): {all_labels.count(i)}")
        print(f"  数据目录: {subject_dir}")
        print(f"  下一步: python collect_real_data.py --train")
        print(f"{'='*50}")

        logger.info("完成! 总段数=%d 集中=%d 一般=%d 涣散=%d",
                    trial_idx, all_labels.count(2), all_labels.count(1), all_labels.count(0))

        # 显示完成画面
        complete_text = visual.TextStim(win, text="采集完成!", height=36, bold=True,
                                         color="#22c55e", units="pix")
        complete_text.draw()
        win.flip()
        core.wait(2)
        win.close()

    ex.register_paradigm("命令行采集 (CLI)", paradigm_loop)

    _orig_close = ex.closeEvent
    ex.closeEvent = lambda: ex.current_win and ex.current_win.close()

    ex.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--task", type=int, default=20, help="集中阶段秒数")
    p.add_argument("--normal", type=int, default=12, help="一般阶段秒数")
    p.add_argument("--rest", type=int, default=12, help="涣散阶段秒数")
    p.add_argument("--fullscr", action="store_true")
    args = p.parse_args()

    run(subject_id=args.subject, n_blocks=args.blocks,
        task_sec=args.task, normal_sec=args.normal, rest_sec=args.rest,
        is_fullscr=args.fullscr)
