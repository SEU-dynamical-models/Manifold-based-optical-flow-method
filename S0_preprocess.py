# -*- coding: utf-8 -*-
# 数据预处理
# Author: Xi Wang
# Date: 28 February 2024
# Email: 2308180834@qq.com

import logging
from pathlib import Path
import json
import re

import mne
import numpy as np
import pandas as pd

# ---------- 配置 ----------
HIGH_FREQ = 100.0   # 低通截止（Hz）
LOW_FREQ = 0.1      # 高通截止（Hz）
BASELINE = (-1.0, -0.1)  # 基线校正窗口（秒）
RESULTS_ROOT = Path(r"/fred/oz284/mc/results/ds004080")
DATA_ROOT = Path(r"/fred/oz284/mc/data/ds004080")
SUBJECTS = ["sub-ccepAgeUMCU07"]  
# SUBJECTS = [name for name in os.listdir(DATA_ROOT) if name.startswith('sub')]
# SUBJECTS = sorted(SUBJECTS)
RUN_RE = re.compile(r"run-(\d{6})")

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("preprocess")

# ---------- 工具函数 ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_runs(data_ieeg_dir: Path) -> list:
    runs = set()
    for root, _, files in os_walk(data_ieeg_dir):
        for fn in files:
            m = RUN_RE.search(fn)
            if m:
                runs.add(m.group(1))
    return sorted(runs)

def os_walk(p: Path):
    for root, dirs, files in __import__("os").walk(str(p)):
        yield root, dirs, files

# ---------- 数据处理流程函数 ----------
def load_json(js_path: Path) -> dict:
    with js_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_raw_brainvision(vhdr_path: Path):
    if not vhdr_path.exists():
        raise FileNotFoundError(f"{vhdr_path} not found")
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True)
    return raw

def preprocess_raw(raw: mne.io.BaseRaw, plf: float):
    # mne.filter 参数为 l_freq (high-pass), h_freq (low-pass)
    raw_filt = raw.copy().filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)
    raw_filt.notch_filter(freqs=plf)
    return raw_filt

def select_channels_from_tsv(raw: mne.io.BaseRaw, channels_tsv: Path) -> mne.io.BaseRaw:
    df = pd.read_csv(channels_tsv, sep="\t")
    cond = (
        (df.get("type") == "ECOG")
        & (df.get("status") == "good")
        & (df.get("status_description") == "included")
    )
    selected = df.loc[cond, "name"].dropna().astype(str).tolist()
    logger.info("Selected channels (%d): %s", len(selected), selected)
    pick_chs = [ch for ch in selected if ch in raw.ch_names]
    if not pick_chs:
        raise RuntimeError("No selected channels found in raw data.")
    return raw.copy().pick_channels(pick_chs)

def rereference(raw: mne.io.BaseRaw):
    # 使用平均参考
    reref, ref_data = mne.set_eeg_reference(raw, ref_channels="average", copy=True)
    return reref

def build_events_from_tsv(events_tsv: Path) -> tuple:
    df = pd.read_csv(events_tsv, sep="\t")
    events_list = []
    event_id = {}
    counter = 0
    for _, row in df.iterrows():
        if row.get("trial_type") != "electrical_stimulation":
            continue
        site = str(row.get("electrical_stimulation_site"))
        onset = int(row.get("sample_start"))
        if site in event_id:
            eid = event_id[site]
        else:
            eid = counter
            event_id[site] = eid
            counter += 1
        events_list.append([onset, 0, eid])
    if not events_list:
        raise RuntimeError("No stimulation events found in TSV.")
    events_arr = np.array(events_list, dtype=int)
    logger.info("Built events: shape=%s, event_id=%s", events_arr.shape, event_id)
    return events_arr, event_id

def make_epochs_and_save(reref_raw: mne.io.BaseRaw, events: np.ndarray, event_id: dict, out_path: Path):
    ensure_dir(out_path.parent)
    epochs = mne.Epochs(
        reref_raw, events, event_id=event_id,
        tmin=-2.0, tmax=3.0,
        baseline=BASELINE,
        preload=True
    )
    epochs.save(str(out_path), overwrite=True)
    logger.info("Saved epochs to %s", out_path)
    return epochs

def save_evokeds_from_epochs(epochs: mne.Epochs, base_run_path: Path, subj: str, ses: str, run_num: str):
    for site_name, eid in epochs.event_id.items():
        # epochs[site_name] 是包含该类型事件的 Epochs 对象
        evoked = epochs[site_name].average()
        # 目录与文件名
        evoked_dir = base_run_path / site_name
        ensure_dir(evoked_dir)
        fname = f"{subj}_{ses}_task-SPESclin_run-{run_num}-{site_name}-ave.fif"
        save_path = base_run_path / fname
        evoked.save(str(save_path), overwrite=True)
        evoked.save(str(evoked_dir / fname), overwrite=True)
        logger.info("Saved evoked for %s to %s", site_name, save_path)

# ---------- 主流程 ----------
def process_subject(subj: str):
    subj_data_dir = DATA_ROOT / subj
    if not subj_data_dir.exists():
        logger.error("Subject data folder not found: %s", subj_data_dir)
        return
    # 第一个子文件夹为session
    ses = next(subj_data_dir.iterdir()).name
    ieeg_dir = subj_data_dir / ses / "ieeg"
    if not ieeg_dir.exists():
        logger.error("iEEG folder not found: %s", ieeg_dir)
        return
    results_subj = RESULTS_ROOT / subj
    ensure_dir(results_subj)

    runs = find_runs(ieeg_dir)
    logger.info("Found runs for %s: %s", subj, runs)

    for run_num in runs:
        run_results_dir = results_subj / f"run-{run_num}"
        ensure_dir(run_results_dir)

        vhdr = ieeg_dir / f"{subj}_{ses}_task-SPESclin_run-{run_num}_ieeg.vhdr"
        js = ieeg_dir / f"{subj}_{ses}_task-SPESclin_run-{run_num}_ieeg.json"
        evts_tsv = ieeg_dir / f"{subj}_{ses}_task-SPESclin_run-{run_num}_events.tsv"
        ch_tsv = ieeg_dir / f"{subj}_{ses}_task-SPESclin_run-{run_num}_channels.tsv"

        try:
            raw = load_raw_brainvision(vhdr)
            jsinfo = load_json(js)
            plf = float(jsinfo.get("PowerLineFrequency", 50.0))
            raw_pre = preprocess_raw(raw, plf)
            picked = select_channels_from_tsv(raw_pre, ch_tsv)
            reref = rereference(picked)
            events_arr, event_id = build_events_from_tsv(evts_tsv)
            epochs_fname = run_results_dir / f"{subj}_{ses}_task-SPESclin_run-{run_num}-epo.fif"
            epochs = make_epochs_and_save(reref, events_arr, event_id, epochs_fname)
            save_evokeds_from_epochs(epochs, run_results_dir, subj, ses, run_num)
        except Exception as e:
            logger.exception("Failed processing %s run %s: %s", subj, run_num, e)
            continue

def main():
    ensure_dir(RESULTS_ROOT)
    for s in SUBJECTS:
        process_subject(s)

if __name__ == "__main__":
    import os
    main()
