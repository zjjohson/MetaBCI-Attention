# -*- coding: utf-8 -*-
"""
MetaBCI 注意力诱导实验 (科技风主题)

通过 Experiment.register_paradigm() 注册到 brainstim 框架。
科技感深色主题界面，心算/阅读/N-back 三种任务模式。

用法:
    c:\...\meta\.venv310\Scripts\python.exe c:\...\meta\brainstim\attention_paradigm.py

Authors: 上大Meta梦  |  License: MIT
"""
import time, random, math

import numpy as _np
for _o, _n in {"alltrue": "all", "sometrue": "any", "product": "prod"}.items():
    if not hasattr(_np, _o): setattr(_np, _o, getattr(_np, _n))
import numpy as np

from psychopy import visual, event, core
from metabci.brainstim.framework import Experiment


def run_attention_experiment(
    n_blocks=6, task_duration=30.0, rest_duration=15.0,
    task_type="math", screen_id=0, is_fullscr=False,
):
    ex = Experiment(
        monitor=None, bg_color_warm=np.array([0.03, 0.04, 0.08]),
        screen_id=screen_id,
        win_size=np.array([1920, 1080] if is_fullscr else [1200, 750]),
        is_fullscr=is_fullscr, record_frames=False,
    )

    # ---- 配色 ----
    A = (0.10, 0.60, 1.0)    # 主色 蓝
    G = (0.90, 0.55, 0.06)   # 辅色 金
    T = (0.95, 0.96, 0.98)    # 主文字
    D = (0.42, 0.48, 0.58)    # 次要文字
    F = (0.18, 0.22, 0.30)    # 浅灰色
    E = (0.22, 0.95, 0.35)    # 绿色

    # ---- 出题 ----
    def _math():
        a, b = random.randint(10, 99), random.randint(1, 99)
        op = random.choice(["+", "-", "*"])
        if op == "+":   return f"{a} + {b} = ?", str(a + b)
        elif op == "-": return f"{a} - {b} = ?", str(a - b)
        else:           return f"{a}  x  {b} = ?", str(a * b)

    _rd = ["注意力是认知资源对特定信息的\n指向与集中，是学习与记忆的基础。",
           "前额叶皮层在注意控制中起着\n核心作用，监控和调节认知过程。",
           "工作记忆的容量约为7个组块\n认知负荷过大会降低学习效率。",
           "多任务切换会额外消耗认知资源\n导致效率显著下降。",
           "正念冥想被证实可以提升\n持续注意力和工作记忆容量。",
           "睡眠质量直接影响前额叶功能\n长期不足显著降低注意力。",
           "哈佛研究: 心智47%时间在走神\n专注当下可提升幸福感。",
           "闭眼时 alpha 波增强\n集中时 beta 波在前额叶活跃。"]
    _ri = -1

    def _next(s):
        if s != "task": return "rest", "", ""
        if task_type == "math": return "math", *_math()
        elif task_type == "reading":
            nonlocal _ri; _ri = (_ri + 1) % len(_rd)
            return "reading", _rd[_ri], ""
        else:
            return "nback", f"字母: {random.choice('ABCDEFGHJKLMNP')}\n与前两个相同?", ""

    def paradigm_loop(win):
        win.setMouseVisible(True)
        W, H = win.size  # pixels
        cw, ch = W / 2, H / 2

        # ============ 科技风装饰元素(一次性创建) ============
        # 四角括号
        corners = []
        m, s = 60, 16  # 边距, 线长
        for (dx, dy) in [(-1, 1), (-1, -1), (1, 1), (1, -1)]:
            x, y = dx * (cw - m), dy * (ch - m + 10)
            h = visual.Rect(win, size=(s, 2), pos=(x - dx * s/2, y), fillColor=A, lineColor=None, units="pix")
            v = visual.Rect(win, size=(2, s), pos=(x, y - dy * s/2), fillColor=A, lineColor=None, units="pix")
            corners.extend([h, v])

        # 角落小方块
        corner_blocks = []
        for (dx, dy) in [(-1, 1), (-1, -1), (1, 1), (1, -1)]:
            corner_blocks.append(visual.Rect(win, size=(6, 6),
                pos=(dx * (cw - m + 3), dy * (ch - m + 13)), fillColor=A, lineColor=None, units="pix"))

        # 顶部横线+标题
        top_line = visual.Rect(win, size=(W - 2*m, 1), pos=(0, ch - m + 30),
                               fillColor=F, lineColor=None, units="pix")
        title_bg = visual.Rect(win, size=(340, 36), pos=(0, ch - m + 15),
                               fillColor=(0.04, 0.05, 0.11), lineColor=F, units="pix", lineWidth=1)
        title_txt = visual.TextStim(win, pos=(0, ch - m + 15), height=18,
                                    color=A, units="pix", bold=True,
                                    text="ATTENTION  PARADIGM")

        # 区块数字框
        block_bg = visual.Rect(win, size=(140, 30), pos=(-cw + m + 80, ch - m + 15),
                               fillColor=None, lineColor=F, units="pix", lineWidth=1)
        block_txt = visual.TextStim(win, pos=(-cw + m + 80, ch - m + 15), height=16,
                                    color=D, units="pix")

        # 状态指示
        status_dot = visual.Circle(win, radius=4, pos=(cw - m - 20, ch - m + 15),
                                   fillColor=A, lineColor=None, units="pix")
        status_txt = visual.TextStim(win, pos=(cw - m - 40, ch - m + 15),
                                     height=16, color=A, units="pix", alignText="right")

        # 倒计时
        timer_txt = visual.TextStim(win, pos=(cw - m - 110, ch - m + 15),
                                    height=16, color=D, units="pix", alignText="right")

        # 中央卡片
        card_w, card_h = 500, 280
        card = visual.Rect(win, size=(card_w, card_h), pos=(0, 0),
                           fillColor=(0.04, 0.05, 0.12), lineColor=F, units="pix", lineWidth=1)
        # 卡片内标题
        card_title = visual.TextStim(win, pos=(0, card_h/2 - 50), height=14,
                                     color=D, units="pix")
        # 卡片内分隔
        card_div = visual.Rect(win, size=(card_w - 60, 1), pos=(0, card_h/2 - 70),
                               fillColor=A, lineColor=None, units="pix")

        # 主要内容
        body = visual.TextStim(win, pos=(0, 10), height=44,
                               color=T, units="pix", bold=True, alignText="center")
        hint = visual.TextStim(win, pos=(0, -card_h/2 + 50), height=18,
                               color=D, units="pix", alignText="center")

        # 进度条
        bar_y = -ch + m - 30
        bar_w = W - 2*m + 40
        prog_bg = visual.Rect(win, size=(bar_w, 2), pos=(0, bar_y),
                              fillColor=F, lineColor=None, units="pix")
        prog = visual.Rect(win, size=(0, 2), pos=(-bar_w/2, bar_y),
                           fillColor=A, lineColor=None, units="pix", anchor="left")

        # 底部轮次块
        dots = []
        for i in range(n_blocks):
            dx = 24 * (i - (n_blocks - 1)/2)
            dots.append(visual.Rect(win, size=(14, 6), pos=(dx, bar_y - 24),
                                    fillColor=F, lineColor=None, units="pix"))

        # 底部信息
        bot_txt = visual.TextStim(win, pos=(0, bar_y - 50), height=13,
                                  color=F, units="pix", text="ESC/Q 退出")

        # 左右扫描线装饰
        scan_lines = []
        for y_off in range(-80, 81, 40):
            alpha = 0.15 - abs(y_off/600)
            scan_lines.append(visual.Rect(win, size=(card_w + 40, 1),
                pos=(0, y_off), fillColor=(*A[:2], alpha/0.6*A[2]), lineColor=None, units="pix"))

        # ---- 主循环 ----
        block, phase, block_start = 0, "task", time.time()
        mode, content, answer = _next("task")
        ct = time.time()
        running, anim_t = True, 0.0

        while running and block < n_blocks:
            anim_t += 0.016
            elapsed = time.time() - block_start
            dur = task_duration if phase == "task" else rest_duration
            remaining = max(0, dur - elapsed)
            pct = min(1.0, elapsed / dur) if dur > 0 else 0

            if remaining <= 0:
                phase = "rest" if phase == "task" else "task"
                if phase == "task": block += 1
                block_start = time.time()
                remaining = dur; pct = 0
                mode, content, answer = _next(phase)
                ct = time.time()

            if phase == "task" and time.time() - ct > 10:
                mode, content, answer = _next("task")
                ct = time.time()

            # 颜色
            ac, sc = (A, G) if phase == "task" else (G, A)
            glow = 0.8 + 0.2 * math.sin(anim_t * 2.0)

            # ---- 绘制 ----
            # 动态扫描线
            for sl in scan_lines:
                sl.fillColor = (*ac[:2], 0.08 * glow * ac[2])
                sl.draw()

            # 角标
            for c in corners:
                c.fillColor = ac; c.draw()
            for cb in corner_blocks:
                cb.fillColor = ac
                cb.size = (6 + 2 * glow, 6 + 2 * glow)
                cb.draw()

            top_line.draw()
            title_bg.draw()
            title_txt.setColor(ac); title_txt.draw()

            block_txt.text = f"[ {block+1} / {n_blocks} ]"; block_txt.draw()
            status_txt.draw()

            m, s = int(remaining // 60), int(remaining % 60)
            timer_txt.text = f"{m:02d} : {s:02d}"; timer_txt.draw()
            status_dot.fillColor = ac; status_dot.draw()

            card.draw()
            card_div.fillColor = ac; card_div.draw()

            if phase == "task":
                status_txt.text = "TASK"
                card_title.text = task_type.upper()
                card_title.color = ac
                card.fillColor = (0.04, 0.05, 0.12)
                body.text = content; body.color = T; body.bold = True
                hint.text = "请心算作答" if task_type == "math" else \
                            "请仔细阅读" if task_type == "reading" else "请判断"
                hint.color = D
            else:
                status_txt.text = "REST"
                card_title.text = "RELAX"
                card_title.color = ac
                card.fillColor = (0.05, 0.04, 0.08)
                body.text = "请放松休息"; body.color = D; body.bold = False
                hint.text = "闭上眼睛 · 保持安静"; hint.color = F

            card_title.draw()
            body.draw()
            hint.draw()

            prog_bg.draw()
            prog.size = (bar_w * pct, 2)
            prog.fillColor = ac; prog.draw()

            for i in range(block):
                dots[i].fillColor = A; dots[i].size = (16, 6)
            for i in range(n_blocks):
                dots[i].draw()

            bot_txt.draw()
            win.flip()

            try:
                for k in event.getKeys():
                    if k in ("escape", "q"):
                        running = False
            except RuntimeError:
                pass

        # 结束
        e1 = visual.TextStim(win, text="EXPERIMENT COMPLETE", pos=(0, 20),
                             height=28, color=E, units="pix", bold=True)
        e2 = visual.TextStim(win, text=f"TOTAL {n_blocks} BLOCKS", pos=(0, -30),
                             height=18, color=D, units="pix")
        e1.draw(); e2.draw(); win.flip(); core.wait(3.0)
        win.close()

    # ====== 注册到框架并通过 ex.run() 运行 ======
    ex.register_paradigm("注意力任务", paradigm_loop)
    # 猴子补丁 closeEvent 防止 core.quit() 导致 pyglet 回调循环
    _orig_close = ex.closeEvent
    ex.closeEvent = lambda: ex.current_win and ex.current_win.close()
    ex.run()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--task", type=float, default=30.0)
    p.add_argument("--rest", type=float, default=15.0)
    p.add_argument("--type", default="math", choices=["math", "reading", "nback"])
    p.add_argument("--fullscr", action="store_true")
    a = p.parse_args()
    run_attention_experiment(a.n_blocks, a.task, a.rest, a.type, is_fullscr=a.fullscr)
