# -*- coding: utf-8 -*-
"""
真实脑电数据采集 + 训练 + PsychoPy 刺激界面

采集时弹出 PsychoPy 全屏界面:
    🟢 集中阶段 — 心算题
    🟡 一般阶段 — 十字注视点
    🔴 涣散阶段 — "闭眼休息"

训练支持: sub-* 和 mema_sub_* 目录

用法:
    python collect_real_data.py --subject 1 --blocks 5    # 采集
    python collect_real_data.py --train                   # 训练
"""
import sys, os, time, json, argparse, logging, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("real_eeg")

REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "attention_dataset", "real_data")
os.makedirs(REAL_DATA_DIR, exist_ok=True)

# ==================== Step 1: 带刺激界面的采集 ====================
def collect_with_paradigm(subject_id=1, n_blocks=5, task_sec=20, normal_sec=12, rest_sec=12, srate=250, fullscreen=False):
    from run_device import NeuracleTCPAdapter
    from attention_dataset.dataset import bandpass_filter

    device = NeuracleTCPAdapter(n_channels=2, srate=srate)
    if not device.connect():
        logger.error("Neuracle 连接失败 (127.0.0.1:8712)")
        return

    # PsychoPy imports
    import numpy as _np
    for _o, _n in {"alltrue": "all", "sometrue": "any", "product": "prod"}.items():
        if not hasattr(_np, _o): setattr(_np, _o, getattr(_np, _n))
    from psychopy import visual, event, core

    import os as _os; _os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # 软件渲染
    win = visual.Window(size=(1920,1080) if fullscreen else (1200,750), fullscr=fullscreen,
                        screen=0, units="pix", color=(0.03,0.04,0.08), colorSpace='rgb',
                        waitBlanking=False, allowGUI=True)
    win.setMouseVisible(True)
    W, H = win.size; cw, ch = W/2, H/2

    BLUE=(0.10,0.60,1.0); GOLD=(0.90,0.55,0.06); WHITE=(0.95,0.96,0.98)
    GRAY=(0.42,0.48,0.58); DIM=(0.18,0.22,0.30); GREEN=(0.22,0.95,0.35)
    RED=(1.0,0.25,0.25); BG_CARD=(0.04,0.05,0.12); m=60

    title = visual.TextStim(win, pos=(0,ch-m+10), height=22, color=BLUE, bold=True, text="ATTENTION  CALIBRATION")
    block_lbl = visual.TextStim(win, pos=(-cw+m+80,ch-m+10), height=16, color=GRAY)
    status_dot = visual.Circle(win, radius=5, pos=(cw-m-30,ch-m+10), fillColor=GREEN, lineColor=None)
    status_txt = visual.TextStim(win, pos=(cw-m-55,ch-m+10), height=16, color=GREEN, alignText="right")
    timer_txt = visual.TextStim(win, pos=(cw-m-130,ch-m+10), height=16, color=GRAY, alignText="right")
    card_w, card_h = 520, 300
    card = visual.Rect(win, size=(card_w,card_h), pos=(0,0), fillColor=BG_CARD, lineColor=DIM, lineWidth=1)
    card_title = visual.TextStim(win, pos=(0,card_h/2-45), height=14, color=GRAY)
    card_div = visual.Rect(win, size=(card_w-60,1), pos=(0,card_h/2-65), fillColor=BLUE, lineColor=None)
    body = visual.TextStim(win, pos=(0,15), height=46, color=WHITE, bold=True, alignText="center")
    hint = visual.TextStim(win, pos=(0,-card_h/2+45), height=18, color=GRAY, alignText="center")
    cross_h = visual.Rect(win, size=(40,3), pos=(0,0), fillColor=GRAY, lineColor=None)
    cross_v = visual.Rect(win, size=(3,40), pos=(0,0), fillColor=GRAY, lineColor=None)
    rest_txt = visual.TextStim(win, pos=(0,0), height=36, color=GRAY, bold=True, text="请闭眼休息")
    bar_y = -ch+m-20; bar_w = W-2*m+40
    prog_bg = visual.Rect(win, size=(bar_w,2), pos=(0,bar_y), fillColor=DIM, lineColor=None)
    prog_bar = visual.Rect(win, size=(0,2), pos=(-bar_w/2,bar_y), fillColor=BLUE, lineColor=None, anchor="left")
    bot = visual.TextStim(win, pos=(0,bar_y-35), height=13, color=DIM, text="ESC 退出")
    dots = [visual.Circle(win, radius=4, pos=(20*(i-(n_blocks-1)/2),bar_y-20), fillColor=DIM, lineColor=None) for i in range(n_blocks)]

    MATH_PROBLEMS = [("43+28","71"),("76-39","37"),("15×6","90"),("52+37","89"),("84-56","28"),
                     ("23×4","92"),("67+19","86"),("91-45","46"),("58+16","74"),("100-63","37")]
    _mi = 0
    def next_math():
        nonlocal _mi; q,a = MATH_PROBLEMS[_mi%len(MATH_PROBLEMS)]; _mi+=1; return q,a

    subject_dir = os.path.join(REAL_DATA_DIR, f"sub-{str(subject_id).zfill(2)}")
    os.makedirs(subject_dir, exist_ok=True)
    trial_idx = 0; all_labels = []
    phases = [("task","集中任务",GREEN,task_sec),("normal","一般状态",GOLD,normal_sec),("rest","涣散休息",RED,rest_sec)]

    clock = core.Clock(); block=0; phase_idx=0; phase_start=clock.getTime()
    running=True; last_switch=clock.getTime(); math_q,math_a=next_math()

    while running and block < n_blocks:
        t = clock.getTime()
        phase_name,phase_label,phase_color,phase_dur = phases[phase_idx]
        elapsed = t-phase_start; remaining=max(0,phase_dur-elapsed); pct=min(1.0,elapsed/phase_dur) if phase_dur else 0

        for k in event.getKeys():
            if k in ("escape","q"): running=False; break

        status_dot.fillColor=phase_color; status_txt.color=phase_color; status_txt.text=phase_label
        mins=int(remaining//60); secs=int(remaining%60); timer_txt.text=f"{mins}:{secs:02d}"
        block_lbl.text=f"BLOCK  {block+1} / {n_blocks}"; prog_bar.size=(bar_w*pct,2)
        for i,d in enumerate(dots): d.fillColor=BLUE if i<block else DIM

        card.autoDraw=card_title.autoDraw=card_div.autoDraw=body.autoDraw=hint.autoDraw=False
        cross_h.autoDraw=cross_v.autoDraw=rest_txt.autoDraw=False

        if phase_name=="task":
            if t-last_switch>10: math_q,math_a=next_math(); last_switch=t
            card_title.text="集中注意力 · 心算练习"
            body.text=math_q; hint.text=f"答案: {math_a}  |  请集中注意力"
            card.autoDraw=card_title.autoDraw=card_div.autoDraw=body.autoDraw=hint.autoDraw=True
        elif phase_name=="normal":
            cross_h.autoDraw=cross_v.autoDraw=True; hint.text="保持自然状态"; hint.autoDraw=True
        else:
            rest_txt.text="请闭眼休息"; rest_txt.autoDraw=True
            hint.text="听到提示音后请睁眼"; hint.autoDraw=True

        win.flip()

        # 采集数据 (与 mema_protocol.py 一致: 1秒窗口, 250Hz, 50%重叠)
        data = device.get_data(n_samples=50)
        if data is not None:
            if not hasattr(collect_with_paradigm,"_buf"): collect_with_paradigm._buf=np.zeros((2,0))
            buf = collect_with_paradigm._buf
            buf = np.concatenate([buf, data.astype(np.float64)], axis=-1)
            # 累积到 1000 点 → 降采样到 250Hz → 250 点 = 1.0s
            while buf.shape[1] >= 1000:
                seg = buf[:, :1000:4]  # 1000Hz → 250Hz
                np.save(os.path.join(subject_dir, f"trial_{trial_idx:04d}.npy"), seg)
                lbl = 2 if phase_name == "task" else (1 if phase_name == "normal" else 0)
                all_labels.append(lbl); trial_idx += 1
                buf = buf[:, 500:]  # 50% 重叠
            collect_with_paradigm._buf = buf

        # 阶段切换
        if remaining<=0:
            try:
                import winsound
                if phases[phase_idx][0]=="rest": winsound.Beep(600,400); time.sleep(0.3); winsound.Beep(800,200)
                elif phases[phase_idx][0]=="task": winsound.Beep(1000,200); time.sleep(0.2); winsound.Beep(1200,200)
                else: winsound.Beep(800,300)
            except: pass
            phase_idx+=1
            if phase_idx>=len(phases): phase_idx=0; block+=1
            phase_start=t; last_switch=t; math_q,math_a=next_math()

    if hasattr(collect_with_paradigm,"_buf"): del collect_with_paradigm._buf
    win.close(); device.close()
    meta={"subject":subject_id,"blocks":block,"labels":all_labels,
         "counts":{"0":all_labels.count(0),"1":all_labels.count(1),"2":all_labels.count(2)}}
    with open(os.path.join(subject_dir,"meta.json"),"w") as f: json.dump(meta,f,indent=2)
    logger.info("="*60); logger.info(f"  完成! 总段数: {trial_idx}")
    logger.info(f"  集中(2):{all_labels.count(2)}, 一般(1):{all_labels.count(1)}, 涣散(0):{all_labels.count(0)}")
    logger.info("="*60)

# ==================== Step 2: 训练 ====================
def train_real_model():
    from attention_dataset.dataset import extract_time_frequency_features, bandpass_filter
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib

    # 匹配 sub-* 和 mema_sub_* 目录
    subj_dirs = []
    for d in os.listdir(REAL_DATA_DIR):
        if (d.startswith("sub-") or d.startswith("mema_sub_")) and os.path.isdir(os.path.join(REAL_DATA_DIR,d)):
            subj_dirs.append(d)

    if not subj_dirs:
        logger.error("未找到数据!")
        return

    print("\n"+"="*60); print(f"  训练 — {len(subj_dirs)} 目录: {subj_dirs}"); print("="*60)

    X_all, y_all = [], []
    for sd in subj_dirs:
        sp = os.path.join(REAL_DATA_DIR, sd)
        files = sorted([f for f in os.listdir(sp) if f.endswith(".npy")])
        meta_path = os.path.join(sp, "meta.json")
        trial_labels = []
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "labels" in meta:
                trial_labels = meta["labels"]
            elif "counts" in meta:
                cnts = meta["counts"]
                for lbl in [0, 1, 2]:
                    trial_labels.extend([lbl] * cnts.get(str(lbl), cnts.get(lbl, 0)))
                trial_labels = trial_labels[:len(files)]

        cnt = 0
        for i, fname in enumerate(files):
            try: seg = np.load(os.path.join(sp, fname))
            except: continue
            if seg.ndim != 2 or seg.shape[0] != 2 or seg.shape[1] < 50: continue
            filtered = bandpass_filter(seg, srate=250, l_freq=0.5, h_freq=45, order=2)
            if filtered.shape[1] < 50: continue
            if filtered.shape[1] > 250: filtered = filtered[:, :250]
            elif filtered.shape[1] < 250: filtered = np.pad(filtered, ((0,0),(0,250-filtered.shape[1])))
            feats = extract_time_frequency_features(filtered[np.newaxis,:,:], srate=250)
            X_all.append(feats[0])
            y_all.append(trial_labels[i] if i < len(trial_labels) else 1)
            cnt += 1
        logger.info(f"  {sd}: {cnt} 段")

    X, y = np.array(X_all), np.array(y_all)
    print(f"\n特征: {X.shape}, 分布: 涣散(0)={sum(y==0)}, 一般(1)={sum(y==1)}, 集中(2)={sum(y==2)}")
    if len(X) < 10:
        logger.error("数据太少!")
        return

    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    X_tr, X_va, y_tr, y_va = train_test_split(X_scaled, y, test_size=0.15, stratify=y, random_state=42)
    clf = MLPClassifier((128,64), max_iter=500, early_stopping=True, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pr = clf.predict(X_va); acc = accuracy_score(y_va, y_pr)
    print(f"\n测试集: {acc:.2%}")
    print(classification_report(y_va, y_pr, target_names=["涣散","一般","集中"]))
    print("混淆矩阵:\n", confusion_matrix(y_va, y_pr))

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "real_classifier.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "real_scaler.pkl"))
    with open(os.path.join(model_dir, "real_config.json"), "w") as f:
        json.dump({"n_subjects": len(subj_dirs), "n_samples": len(y), "accuracy": float(acc)}, f, indent=2)
    print(f"\n✓ 模型已保存 → models/real_*.pkl")
    print(f"  现在 run_device.py --device neuracle_tcp 自动使用真实模型")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--task", type=int, default=20)
    p.add_argument("--normal", type=int, default=12)
    p.add_argument("--rest", type=int, default=12)
    args = p.parse_args()

    if args.train:
        train_real_model()
    else:
        collect_with_paradigm(subject_id=args.subject, n_blocks=args.blocks,
                              task_sec=args.task, normal_sec=args.normal, rest_sec=args.rest,
                              fullscreen=args.fullscreen)
