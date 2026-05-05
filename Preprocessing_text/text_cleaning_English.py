import re
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Extracting data"))
from collection import JSONLCombiner

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_DIR  = os.path.join(SCRIPT_DIR, "..", "Extracting data", "jsonl_files")
OUT_DIR    = os.path.join(SCRIPT_DIR, "cleaned")
os.makedirs(OUT_DIR, exist_ok=True)

# Non-WLS sources — eligible for train/test split
non_wls_files = [
    os.path.join(JSONL_DIR, "English_Pitt_Control_cookie_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Pitt_Dementia_cookie_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Lu_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Baycrest_output.jsonl"),
    os.path.join(JSONL_DIR, "English_VAS_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Kempler_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Delaware_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_taukadial_train_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_taukadial_test_output.jsonl"),
]
# WLS is training-only per paper Section 3.1.4
wls_files = [os.path.join(JSONL_DIR, "English_WLS_output.jsonl")]

combiner = JSONLCombiner(non_wls_files + wls_files, OUT_DIR, "combined_jsonl_English.jsonl")
combiner.combine()

non_wls_df = pd.concat(
    [pd.read_json(f, lines=True) for f in non_wls_files if os.path.exists(f)],
    ignore_index=True
)
wls_df = pd.read_json(wls_files[0], lines=True)


def remove_zh_language_rows(df):
    return df[df["Languages"] != "zh"]


def clean_diagnosis(df):
    diagnoses_to_remove = ["Vascular", "Memory", "Aphasia", "Pick's", "Other"]
    df = df[~df["Diagnosis"].isin(diagnoses_to_remove)]
    df = df[df["Diagnosis"].notna() & (df["Diagnosis"] != "")]
    df["Diagnosis"] = df["Diagnosis"].replace({
        "Control": "HC",
        "Conrol":  "HC",
        "NC":      "HC",
        "H":       "HC",
        "AD":      "Dementia",
        "PossibleAD":        "Dementia",
        "ProbableAD":        "Dementia",
        "potential dementia":"Dementia",
        "D":                 "Dementia",
        "Alzheimer's":       "Dementia",
    })
    return df


def preprocess_text(text):
    text = re.sub(r"\b[A-Z]{3}\b", "", text)
    text = re.sub(r"xxx", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.replace("PAR", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\\x[0-9A-Za-z_]+\\x", "", text)
    text = re.sub(r"\b\w+:\s*", "", text)
    text = text.replace("\n", " ")
    text = text.replace("→", "")
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r'[\\+^"/„]', "", text)
    text = re.sub(r"[_']", "", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("&=laughs", "")
    text = text.replace("&=nods", "")
    text = text.replace("&=coughs", "")
    text = text.replace("&=snaps:tongue", "")
    text = text.replace("<", "").replace(">", "")
    text = text.replace("*", "").replace("&", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.,!?;:])\s+\1", r"\1", text)
    text = re.sub(r"(\.\s*){2,}", ".", text)
    if "." in text:
        text = text.rsplit(".", 1)[0] + "."
    return text


def process(df):
    df = remove_zh_language_rows(df)
    df = clean_diagnosis(df)
    df["Text_interviewer_participant"] = df["Text_interviewer_participant"].apply(preprocess_text)
    df["Text_length"] = df["Text_interviewer_participant"].apply(len)
    return df[df["Text_length"] > 60].reset_index(drop=True)


non_wls_df = process(non_wls_df)
wls_df = process(wls_df)

# 80/20 split on non-WLS only; all WLS goes to training
train_base, test_en = train_test_split(
    non_wls_df, test_size=0.2, stratify=non_wls_df["Diagnosis"], random_state=42
)
train_en = pd.concat([train_base, wls_df], ignore_index=True)

train_en.to_json(os.path.join(OUT_DIR, "train_english.jsonl"), orient="records", lines=True, force_ascii=False)
test_en.to_json(os.path.join(OUT_DIR,  "test_english.jsonl"),  orient="records", lines=True, force_ascii=False)

print(f"Non-WLS: {len(non_wls_df)} records  WLS: {len(wls_df)} records")
print(f"Train: {len(train_en)} (non-WLS {len(train_base)} + WLS {len(wls_df)})  Test: {len(test_en)}")
print("Train diagnosis dist:", dict(train_en["Diagnosis"].value_counts()))
print("Test  diagnosis dist:", dict(test_en["Diagnosis"].value_counts()))
