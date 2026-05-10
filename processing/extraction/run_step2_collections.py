"""
Step 2: Data Collection & Extraction
Runs all collection scripts, attaches labels to ASR datasets, and writes
JSONL output to data/processed/extracted/.
Run from the repo root: python processing/extraction/run_step2_collections.py
"""
import csv
import os
import sys
import json
from dataclasses import asdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collection import JSONLCombiner
from cha_collection import CHACollection
from TSV_collection import TSVCollection
from ASR_collection import ASRCollection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
JSONL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "extracted")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
os.makedirs(JSONL_DIR, exist_ok=True)

TRANS = os.path.join(PROJECT_ROOT, "data", "processed", "transcriptions")


def _first_existing_path(*paths: str) -> str | None:
    """Return the first existing path from the candidates, else None."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


# ── Label tables for ASR datasets ────────────────────────────────────────────

DEMCARE = os.path.join(DATA_ROOT, "Greek", "Dem@Care")


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
    path = os.path.join(DATA_ROOT, "ADReSS-M", "test-gr-groundtruth.csv")
    labels = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["addressfname"]] = row
    return labels


def _load_greek_labels() -> dict:
    """PatientCode (str) → {Dx, AGE, EDU, MMSE} — greek-groundtruth.csv (DS5 metadata)."""
    path = os.path.join(DATA_ROOT, "Greek", "greek-groundtruth.csv")
    labels = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["PatientCode"]] = row
    return labels


def _load_taukadial_labels() -> dict:
    """tkdname stem (no .wav) → {dx, mmse, age, sex} for train + test."""
    labels = {}
    train_path = os.path.join(DATA_ROOT, "TAUKADIAL", "TAUKADIAL-24-train", "TAUKADIAL-24", "train", "groundtruth.csv")
    with open(train_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = row["tkdname"].replace(".wav", "")
            labels[stem] = row
    # test groundtruth is semicolon-delimited
    test_gt_path = os.path.join(DATA_ROOT, "TAUKADIAL", "testgroundtruth.csv")
    test_meta_path = os.path.join(DATA_ROOT, "TAUKADIAL", "meta_test.csv")
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


def _load_wls_labels() -> dict:
    """
    file_id_int -> {Diagnosis, Age, Gender, Education} for WLS records.
    Default path reproduces the paper's effective WLS weak-label rule:
    participants meeting the 2011 category fluency threshold are HC;
    all remaining spreadsheet rows are treated as Dementia-side cases.
    Threshold: age<60->16, 60-79->14, >79->12 words (paper Section 3.1.4).
    Join key: idtlkbnk % 100000 == int(File_ID).
    """
    xlsx = os.path.join(DATA_ROOT, "English", "WLS", "WLS-data.xlsx")
    labels = {}
    xl_2011 = pd.read_excel(xlsx, sheet_name="Data - 2004, 2011")
    xl_2011["file_id"] = xl_2011["idtlkbnk"].astype(int) % 100000

    # Paper-style weak labels: non-HC rows remain on the impaired side even
    # when the 2011 category-fluency value is missing/refused.
    fluency_col = "category fluency, scored words named, 2011"
    age_col = "age 2011"
    for _, row in xl_2011.iterrows():
        fid = int(row["file_id"])
        score = row.get(fluency_col)
        age = row.get(age_col)
        diagnosis = "Dementia"
        if not (pd.isna(score) or score <= 0 or pd.isna(age) or age <= 0):
            threshold = 16 if age < 60 else (14 if age <= 79 else 12)
            if score >= threshold:
                diagnosis = "HC"
        labels[fid] = {
            "Diagnosis": diagnosis,
            "Age": str(int(age)) if not pd.isna(age) and age > 0 else "Unknown",
            "Gender": "",
            "Education": "",
        }

    return labels


def _load_vas_labels() -> dict:
    """
    file_id_int -> {Diagnosis, Age, Gender, Education, Moca} for VAS records.
    Source: data/English/VAS/demo.xlsx (preferred) or 0demo.xlsx.
    """
    base = os.path.join(DATA_ROOT, "English", "VAS")
    labels = {}
    df = None
    for fname in ("demo.xlsx", "0demo.xlsx"):
        path = os.path.join(base, fname)
        if os.path.exists(path):
            df = pd.read_excel(path)
            break
    if df is None:
        return labels

    for _, row in df.iterrows():
        vas_id = row.get("VAS ID")
        dx = row.get("H/MCI/D")
        if pd.isna(vas_id) or pd.isna(dx):
            continue
        try:
            fid = int(float(vas_id))
        except (ValueError, TypeError):
            continue

        dx = str(dx).strip()
        if dx not in {"H", "MCI", "D"}:
            continue

        age = row.get("age", "Unknown")
        gender = row.get("gender", "")
        education = row.get("Highest Education", "")
        moca = row.get("moca", row.get("MOCA", "Unknown"))

        labels[fid] = {
            "Diagnosis": dx,
            "Age": str(int(age)) if not pd.isna(age) else "Unknown",
            "Gender": str(gender).strip() if not pd.isna(gender) else "",
            "Education": str(education).strip() if not pd.isna(education) else "",
            "Moca": str(int(moca)) if not pd.isna(moca) and str(moca).strip() != "" else "Unknown",
        }

    return labels


def _load_kempler_labels() -> dict:
    """
    Kempler file stem -> metadata.
    All Kempler cases are AD patients per corpus note provided by user.
    d6cookie is the Cookie Theft variant of participant d6.
    """
    base = {
        "d1": {"Age": "74", "Education": "11 yrs", "MMSE": "Unknown", "Gender": "M"},
        "d4": {"Age": "82", "Education": "BA", "MMSE": "15", "Gender": "F"},
        "d5": {"Age": "Unknown", "Education": "Unknown", "MMSE": "6", "Gender": "M"},
        "d6": {"Age": "87", "Education": "Unknown", "MMSE": "15", "Gender": "F"},
        "d9": {"Age": "65", "Education": "11 yrs", "MMSE": "14", "Gender": "M"},
        "d10": {"Age": "82", "Education": "8th grade", "MMSE": "17", "Gender": "M"},
    }
    labels = {}
    for stem, meta in base.items():
        labels[stem] = {"Diagnosis": "AD", **meta}
    labels["d6cookie"] = {"Diagnosis": "AD", **base["d6"]}
    return labels


def _load_ncmmsc_labels() -> dict:
    """
    Stem → {Diagnosis, Gender} for NCMMSC2021_AD long-audio dataset.
    Labels are encoded in filenames: {DX}_{Gender}_{PID}_{Session}.wav
    Covers both train/{AD,HC,MCI}/ and test_have_label/ (test_none_label skipped).
    """
    labels = {}
    base = os.path.join(DATA_ROOT, "NCMMSC2021_AD", "AD_dataset_long")
    sources = [
        os.path.join(base, "train", "AD"),
        os.path.join(base, "train", "HC"),
        os.path.join(base, "train", "MCI"),
        os.path.join(base, "test_have_label"),
    ]
    for folder in sources:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(".wav"):
                continue
            stem = os.path.splitext(fname)[0]
            parts = stem.split("_")
            if len(parts) >= 2 and parts[0] in ("AD", "HC", "MCI"):
                labels[stem] = {"Diagnosis": parts[0], "Gender": parts[1]}
    return labels


_DS7_LABELS       = _load_demcare_labels("long")
_DS5_LABELS       = _load_demcare_labels("short")
_DS3_PILOT        = _load_demcare_pilot_labels()
_GREEK_META       = _load_greek_labels()   # MMSE/age/edu metadata for DS5 patients
_ADRESS_M_GR      = _load_adress_m_gr_labels()
_TAUKADIAL_LABELS = _load_taukadial_labels()
_WLS_LABELS       = _load_wls_labels()
_VAS_LABELS       = _load_vas_labels()
_KEMPLER_LABELS   = _load_kempler_labels()
_NCMMSC_LABELS    = _load_ncmmsc_labels()


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


def _attach_wls_labels(record: dict) -> bool:
    """
    Join WLS labels on File_ID integer.
    Priority: 2020 research consensus > 2011 category fluency threshold.
    """
    try:
        fid = int(record.get("File_ID", ""))
    except (ValueError, TypeError):
        return False
    meta = _WLS_LABELS.get(fid)
    if meta:
        record["Diagnosis"] = meta["Diagnosis"]
        if meta.get("Age"):
            record["Age"] = meta["Age"]
        if meta.get("Gender"):
            record["Gender"] = meta["Gender"]
        return True
    return False


def _attach_vas_labels(record: dict) -> bool:
    """Join VAS labels on integer File_ID from demo.xlsx / 0demo.xlsx."""
    try:
        fid = int(record.get("File_ID", ""))
    except (ValueError, TypeError):
        return False
    meta = _VAS_LABELS.get(fid)
    if meta:
        record["Diagnosis"] = meta["Diagnosis"]
        record["Age"] = meta["Age"]
        record["Gender"] = meta["Gender"]
        record["Education"] = meta["Education"]
        record["Moca"] = meta["Moca"]
        return True
    return False


def _attach_ncmmsc_labels(record: dict) -> bool:
    """
    Join NCMMSC labels on the filename stem.
    File_ID from ASRCollection is the relative path (e.g. train/AD/AD_F_030807_001);
    take the last path component as the stem.
    """
    stem = record.get("File_ID", "").replace("\\", "/").split("/")[-1]
    meta = _NCMMSC_LABELS.get(stem)
    if meta:
        record["Diagnosis"] = meta["Diagnosis"]
        record["Gender"]    = meta["Gender"]
        record["Dataset"]   = "NCMMSC2021_AD"
        return True
    return False


def _attach_kempler_labels(record: dict) -> bool:
    """Attach Kempler labels and available demographics from corpus note."""
    stem = str(record.get("File_ID", "")).strip().lower()
    meta = _KEMPLER_LABELS.get(stem)
    if meta:
        record["Diagnosis"] = meta["Diagnosis"]
        record["Age"] = meta["Age"]
        record["Gender"] = meta["Gender"]
        record["Education"] = meta["Education"]
        record["MMSE"] = meta["MMSE"]
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
        path = os.path.join(DATA_ROOT, "English", "Pitt", group, "cookie")
        name = f"English_Pitt_{group}_cookie_output.jsonl"
        produced.append(run_cha(path, "english", name))

    for ds in ("Lu", "Baycrest", "Delaware", "Kempler", "VAS", "WLS"):
        path = os.path.join(DATA_ROOT, "English", ds)
        name = f"English_{ds}_output.jsonl"
        out = run_cha(path, "english", name)
        if ds == "Kempler":
            n_labeled = _attach_labels_to_jsonl(out, _attach_kempler_labels)
            print(f"    → {n_labeled} Kempler records labeled from corpus note")
        if ds == "VAS":
            n_labeled = _attach_labels_to_jsonl(out, _attach_vas_labels)
            print(f"    → {n_labeled} VAS records labeled from demo.xlsx")
        if ds == "WLS":
            n_labeled = _attach_labels_to_jsonl(out, _attach_wls_labels)
            print(f"    → {n_labeled} WLS records labeled (2020 consensus + 2011 fluency threshold)")
        produced.append(out)

    # ── Spanish CHA datasets ──────────────────────────────────────────────────
    print("\n[Spanish CHA]")
    for ds in ("Ivanova", "PerLA"):
        path = os.path.join(DATA_ROOT, "Spanish", ds)
        name = f"Spanish_{ds}_output.jsonl"
        produced.append(run_cha(path, "spanish", name))

    # ── Chinese TSV dataset ───────────────────────────────────────────────────
    print("\n[Chinese TSV]")
    produced.append(run_tsv(os.path.join(DATA_ROOT, "Chinese", "iFlytek"), "Chinese_iFlytek_output.jsonl"))

    # ── Chinese ASR (NCMMSC2021) ──────────────────────────────────────────────
    print("\n[Chinese ASR]")
    ncmmsc_src = _first_existing_path(
        os.path.join(TRANS, "ncmmsc_transcriptions.jsonl"),
        os.path.join(TRANS, "ncmmsc_transcriptions.json"),
    )
    if ncmmsc_src:
        out = run_asr(ncmmsc_src, "Chinese_NCMMSC_output.jsonl")
        n_labeled = _attach_labels_to_jsonl(out, _attach_ncmmsc_labels)
        print(f"    → {n_labeled} NCMMSC records labeled from filename")
        produced.append(out)
    else:
        print("  ncmmsc_transcriptions.jsonl not found — run ASR_audio_dataset.py --dataset ncmmsc first")

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
