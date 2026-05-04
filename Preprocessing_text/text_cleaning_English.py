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

input_files = [
    os.path.join(JSONL_DIR, "English_Pitt_Control_cookie_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Pitt_Dementia_cookie_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Lu_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Baycrest_output.jsonl"),
    os.path.join(JSONL_DIR, "English_VAS_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Kempler_output.jsonl"),
    os.path.join(JSONL_DIR, "English_WLS_output.jsonl"),
    os.path.join(JSONL_DIR, "English_Delaware_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_taukadial_train_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_taukadial_test_output.jsonl"),
]
output_filename = "combined_jsonl_English.jsonl"
combiner = JSONLCombiner(input_files, OUT_DIR, output_filename)
combiner.combine()
English_df = pd.read_json(os.path.join(OUT_DIR, output_filename), lines=True)


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


English_df = remove_zh_language_rows(English_df)
English_df = clean_diagnosis(English_df)
English_df["Text_interviewer_participant"] = English_df["Text_interviewer_participant"].apply(preprocess_text)
English_df["Text_length"] = English_df["Text_interviewer_participant"].apply(len)


def remove_short_transcripts(df, min_length=60):
    return df[df["Text_length"] > min_length]


English_df = remove_short_transcripts(English_df)
train_en, test_en = train_test_split(
    English_df, test_size=0.2, stratify=English_df["Diagnosis"], random_state=42
)

train_en.to_json(os.path.join(OUT_DIR, "train_english.jsonl"), orient="records", lines=True, force_ascii=False)
test_en.to_json(os.path.join(OUT_DIR,  "test_english.jsonl"),  orient="records", lines=True, force_ascii=False)

print(f"English: {len(English_df)} records after cleaning")
print(English_df["Diagnosis"].value_counts().to_string())
print(f"Train: {len(train_en)}  Test: {len(test_en)}")
