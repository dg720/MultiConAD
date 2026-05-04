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
    os.path.join(JSONL_DIR, "ASR_adress_m_gr_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_ds3_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_ds5_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_ds7_output.jsonl"),
]
output_filename = "combined_jsonl_Greek_M_ADReSS_D3_5_7.jsonl"
combiner = JSONLCombiner(input_files, OUT_DIR, output_filename)
combiner.combine()


def preprocess_text_audio_data(text, TF_IDF=False):
    text = re.sub(r"Speaker \d+:", "", text).strip()
    text = re.sub(r"\.{2,}", "", text)
    if TF_IDF:
        text = re.sub(r"[^\w\s]", "", text)
    return text


def remove_short_transcripts(df, min_length=60):
    return df[df["length"] >= min_length]


Greek_df = pd.read_json(os.path.join(OUT_DIR, output_filename), lines=True)
Greek_df["Diagnosis"] = Greek_df["Diagnosis"].replace({
    "Control":    "HC",
    "healthy":    "HC",
    "HEALTHY":    "HC",
    "ProbableAD": "AD",
    "MCI":        "MCI",
    "ΜCI":        "MCI",
})

Greek_df = Greek_df[Greek_df["Diagnosis"] != "Unknown"]
Greek_df["Text_interviewer_participant"] = Greek_df["Text_interviewer_participant"].apply(preprocess_text_audio_data)
Greek_df["length"] = Greek_df["Text_interviewer_participant"].apply(len)
Greek_df = remove_short_transcripts(Greek_df)

train_gr, test_gr = train_test_split(
    Greek_df, test_size=0.2, stratify=Greek_df["Diagnosis"], random_state=42
)

train_gr.to_json(os.path.join(OUT_DIR, "train_greek.jsonl"), orient="records", lines=True, force_ascii=False)
test_gr.to_json(os.path.join(OUT_DIR,  "test_greek.jsonl"),  orient="records", lines=True, force_ascii=False)

print(f"Greek: {len(Greek_df)} records after cleaning")
print(Greek_df["Diagnosis"].value_counts().to_string())
print(f"Train: {len(train_gr)}  Test: {len(test_gr)}")
