"""
Step 2: Data Collection & Extraction
Runs all collection scripts, attaches labels to ASR datasets, and writes
JSONL output to jsonl_files/.
Run from the 'Extracting data/' directory:  python run_step2_collections.py
"""
import csv
import os
import sys
import json
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collection import JSONLCombiner
from cha_collection import CHACollection
from TSV_collection import TSVCollection
from ASR_collection import ASRCollection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
JSONL_DIR = os.path.join(SCRIPT_DIR, "jsonl_files")
RAW = os.path.join(PROJECT_ROOT, "raw_datasets")
os.makedirs(JSONL_DIR, exist_ok=True)

DATA = os.path.join(PROJECT_ROOT, "data")
TRANS = os.path.join(PROJECT_ROOT, "transcriptions")


# ── Label tables for ASR datasets ────────────────────────────────────────────

DEMCARE = os.path.join(RAW, "Greek", "Dem@Care")


def _load_demcare_labels(protocol: str) -> dict:
    """
    Build patient_id (str) → diagnosis from Dem@Care 0tasks directories.
    Uses 0tasks/ as primary source (more complete than .cha dirs).
    protocol: "long" (DS7) | "short" (DS5)
    """
    labels = {}
    base = os.path.join(DEMCARE, protocol, "0tasks")
    for dx in ("AD", "HC", "MCI"):
        folder = os.path.join(base, dx)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            pid = os.path.splitext(fname)[0].replace("tasks", "")
            labels[pid] = dx
    return labels


def _load_demcare_pilot_labels() -> dict:
    """
    DS3 pilot: patient_id (str) → diagnosis.
    control/ → HC, patients/ → AD (pilot has no MCI sub-group).
    IDs 1-10 overlap between control and patients; control takes priority.
    """
    labels = {}
    base = os.path.join(DEMCARE, "pilot")
    for pid in os.listdir(os.path.join(base, "patients")):
        labels[pid] = "AD"
    for pid in os.listdir(os.path.join(base, "control")):
        labels[pid] = "HC"   # overwrite if overlap
    return labels


def _load_adress_m_gr_labels() -> dict:
    """addressfname (str) → {dx, age, gender, educ, mmse} for ADReSS-M test-gr."""
    path = os.path.join(RAW, "ADReSS-M", "test-gr-groundtruth.csv")
    labels = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["addressfname"]] = row
    return labels


def _load_greek_labels() -> dict:
    """PatientCode (str) → {Dx, AGE, EDU, MMSE} — greek-groundtruth.csv (DS5 metadata)."""
    path = os.path.join(RAW, "Greek", "greek-groundtruth.csv")
    labels = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["PatientCode"]] = row
    return labels


def _load_taukadial_labels() -> dict:
    """tkdname stem (no .wav) → {dx, mmse, age, sex} for train + test."""
    labels = {}
    train_path = os.path.join(RAW, "TAUKADIAL", "TAUKADIAL-24-train", "TAUKADIAL-24", "train", "groundtruth.csv")
    with open(train_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = row["tkdname"].replace(".wav", "")
            labels[stem] = row
    # test groundtruth is semicolon-delimited
    test_gt_path = os.path.join(RAW, "TAUKADIAL", "testgroundtruth.csv")
    test_meta_path = os.path.join(RAW, "TAUKADIAL", "meta_test.csv")
    test_gt, test_meta = {}, {}
    with open(test_gt_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("tkdname"):
                continue
            parts = line.split(";")
            if len(parts) >= 3:
                stem = parts[0].replace(".wav", "")
                test_gt[stem] = {"mmse": parts[1], "dx": parts[2]}
    with open(test_meta_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("tkdname"):
                continue
            parts = line.split(";")
            if len(parts) >= 3:
                stem = parts[0].replace(".wav", "")
                test_meta[stem] = {"age": parts[1], "sex": parts[2]}
    for stem, gt in test_gt.items():
        meta = test_meta.get(stem, {})
        labels[stem] = {**gt, **meta, "tkdname": stem + ".wav"}
    return labels


_DS7_LABELS       = _load_demcare_labels("long")
_DS5_LABELS       = _load_demcare_labels("short")
_DS3_PILOT        = _load_demcare_pilot_labels()
_GREEK_META       = _load_greek_labels()   # MMSE/age/edu metadata for DS5 patients
_ADRESS_M_GR      = _load_adress_m_gr_labels()
_TAUKADIAL_LABELS = _load_taukadial_labels()


def _attach_labels_to_jsonl(path: str, attach_fn) -> int:
    """Read JSONL, apply attach_fn to each record dict, rewrite in place."""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    updated = 0
    for r in records:
        if attach_fn(r):
            updated += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    return updated


def _attach_adress_m_gr_labels(record: dict) -> bool:
    """Join test-gr-groundtruth on addressfname from File_ID."""
    meta = _ADRESS_M_GR.get(record.get("File_ID", ""))
    if meta:
        record["Diagnosis"] = meta["dx"]
        record["Age"]       = meta["age"]
        record["Gender"]    = meta["gender"]
        record["Education"] = meta["educ"]
        record["MMSE"]      = meta["mmse"]
        return True
    return False


def _attach_ds7_labels(record: dict) -> bool:
    """Join Dem@Care long/0tasks on PatientNNN from File_ID."""
    pid = record.get("File_ID", "").replace("Patient", "").strip()
    dx = _DS7_LABELS.get(pid)
    if dx:
        record["Diagnosis"] = dx
        return True
    return False


def _attach_ds5_labels(record: dict) -> bool:
    """Join Dem@Care short/0tasks on PatientNNN; also attach MMSE/age/edu from greek-groundtruth."""
    pid = record.get("File_ID", "").replace("Patient", "").strip()
    dx = _DS5_LABELS.get(pid)
    if dx:
        record["Diagnosis"] = dx
        meta = _GREEK_META.get(pid, {})
        if meta:
            record["Age"]       = meta.get("AGE", record.get("Age", "Unknown"))
            record["Education"] = meta.get("EDU", record.get("Education", "Unknown"))
            record["MMSE"]      = meta.get("MMSE", record.get("MMSE", "Unknown"))
        return True
    return False


def _attach_ds3_labels(record: dict) -> bool:
    """
    Dem@Care pilot for 'patient N' dirs; directory-name inference for controlday* dirs.
    """
    parts = record.get("File_ID", "").replace("\\", "/").split("/")
    top = parts[0].lower()
    # controlday* directories → HC
    if top.startswith("control"):
        record["Diagnosis"] = "HC"
        return True
    # 'patient N' directories → look up pilot label by N
    if len(parts) > 1:
        patient_dir = parts[1].lower()
        if patient_dir.startswith("patient "):
            pid = patient_dir.replace("patient ", "").strip()
            dx = _DS3_PILOT.get(pid)
            if dx:
                record["Diagnosis"] = dx
                return True
    return False


def _attach_taukadial_labels(record: dict) -> bool:
    """Join TAUKADIAL groundtruth on File_ID stem."""
    stem = record.get("File_ID", "")
    meta = _TAUKADIAL_LABELS.get(stem)
    if meta:
        record["Diagnosis"] = meta.get("dx", "Unknown")
        record["Age"] = meta.get("age", record.get("Age", "Unknown"))
        record["Gender"] = meta.get("sex", meta.get("sex", record.get("Gender", "Unknown")))
        record["MMSE"] = meta.get("mmse", record.get("MMSE", "Unknown"))
        return True
    return False


# ── Collection runners ────────────────────────────────────────────────────────

def run_cha(path: str, language: str, output_name: str) -> str:
    out = os.path.join(JSONL_DIR, output_name)
    col = CHACollection(path, language=language)
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for dp in col.get_normalized_data():
            json.dump(asdict(dp), f, ensure_ascii=False)
            f.write("\n")
            count += 1
    print(f"  {output_name}: {count} records")
    return out


def run_tsv(path: str, output_name: str) -> str:
    out = os.path.join(JSONL_DIR, output_name)
    col = TSVCollection(path, language="chinese")
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for dp in col.get_normalized_data():
            json.dump(asdict(dp), f, ensure_ascii=False)
            f.write("\n")
            count += 1
    print(f"  {output_name}: {count} records")
    return out


def run_asr(json_path: str, output_name: str) -> str:
    out = os.path.join(JSONL_DIR, output_name)
    col = ASRCollection(json_path)
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for dp in col.get_normalized_data():
            json.dump(asdict(dp), f, ensure_ascii=False)
            f.write("\n")
            count += 1
    print(f"  {output_name}: {count} records")
    return out


if __name__ == "__main__":
    produced = []

    # ── English CHA datasets ──────────────────────────────────────────────────
    print("\n[English CHA]")
    for group in ("Control", "Dementia"):
        path = os.path.join(DATA, "English", "Pitt", group, "cookie")
        name = f"English_Pitt_{group}_cookie_output.jsonl"
        produced.append(run_cha(path, "english", name))

    for ds in ("Lu", "Baycrest", "Delaware", "Kempler", "VAS", "WLS"):
        path = os.path.join(DATA, "English", ds)
        name = f"English_{ds}_output.jsonl"
        produced.append(run_cha(path, "english", name))

    # ── Spanish CHA datasets ──────────────────────────────────────────────────
    print("\n[Spanish CHA]")
    for ds in ("Ivanova", "PerLA"):
        path = os.path.join(DATA, "Spanish", ds)
        name = f"Spanish_{ds}_output.jsonl"
        produced.append(run_cha(path, "spanish", name))

    # ── Chinese TSV dataset ───────────────────────────────────────────────────
    print("\n[Chinese TSV]")
    produced.append(run_tsv(os.path.join(DATA, "Chinese", "iFlytek"), "Chinese_iFlytek_output.jsonl"))

    # ── ASR (Greek + TAUKADIAL) ───────────────────────────────────────────────
    print("\n[ASR transcriptions]")
    asr_datasets = [
        ("adress_m_gr_transcriptions.json",    "ASR_adress_m_gr_output.jsonl",    _attach_adress_m_gr_labels),
        ("ds3_transcriptions.json",            "ASR_ds3_output.jsonl",            _attach_ds3_labels),
        ("ds5_transcriptions.json",            "ASR_ds5_output.jsonl",            _attach_ds5_labels),
        ("ds7_transcriptions.json",            "ASR_ds7_output.jsonl",            _attach_ds7_labels),
        ("taukadial_test_transcriptions.json", "ASR_taukadial_test_output.jsonl", _attach_taukadial_labels),
        ("taukadial_train_transcriptions.json","ASR_taukadial_train_output.jsonl",_attach_taukadial_labels),
    ]
    for src, name, label_fn in asr_datasets:
        out = run_asr(os.path.join(TRANS, src), name)
        if label_fn:
            n_labeled = _attach_labels_to_jsonl(out, label_fn)
            print(f"    → {n_labeled} records labeled")
        produced.append(out)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDone. {len(produced)} JSONL files written to {JSONL_DIR}")
    total = 0
    for p in produced:
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        total += n
        print(f"  {os.path.basename(p)}: {n}")
    print(f"\nTotal records: {total}")
