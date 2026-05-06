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

_ncmmsc_path = os.path.join(JSONL_DIR, "Chinese_NCMMSC_output.jsonl")
input_files = [
    os.path.join(JSONL_DIR, "Chinese_iFlytek_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_taukadial_train_output.jsonl"),
    os.path.join(JSONL_DIR, "ASR_taukadial_test_output.jsonl"),
]
if os.path.exists(_ncmmsc_path):
    input_files.append(_ncmmsc_path)
output_filename = "combined_jsonl_Chinese_iFlyTek_Taukdial_NCMMSC.jsonl"
combiner = JSONLCombiner(input_files, OUT_DIR, output_filename)
combiner.combine()


def remove_english_rows(df):
    return df[df["Languages"] != "en"]


def preprocess_text(text):
    text = text.replace("//", "").replace("/", "")
    text = re.sub(r"\s*&\w+", "", text).strip()
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("Doctor:", "").replace("Patient:", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text_audio_data(text):
    text = re.sub(r"Speaker \d+:", "", text).strip()
    return text


def preprocess_dataset(df, dataset_name):
    if dataset_name in df["Dataset"].values:
        mask = df["Dataset"] == dataset_name
        df.loc[mask, "Text_interviewer_participant"] = (
            df.loc[mask, "Text_interviewer_participant"].apply(preprocess_text)
        )
    return df


def preprocess_dataset_audio(df):
    for index, row in df.iterrows():
        if row["Dataset"] != "Predictive_Chinese_challenge_Chinese_2019":
            df.at[index, "Text_interviewer_participant"] = preprocess_text_audio_data(
                row["Text_interviewer_participant"]
            )
    return df


def filter_by_length(row, stats):
    mean = stats.loc[row["Diagnosis"], "mean"]
    std  = stats.loc[row["Diagnosis"], "std"]
    return mean - std <= row["length"] <= mean + std


df_chinese = pd.read_json(os.path.join(OUT_DIR, output_filename), lines=True)
print(f"Input records: {len(df_chinese)}  (datasets: {df_chinese['Dataset'].value_counts().to_dict()})")

# Drop iFlytek test set (no labels) and any remaining Unknown
df_chinese = df_chinese[df_chinese["Diagnosis"] != "Unknown"]

df_chinese["Diagnosis"] = df_chinese["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})
df_chinese = remove_english_rows(df_chinese)

df_chinese = preprocess_dataset(df_chinese, "Predictive_Chinese_challenge_Chinese_2019")
df_chinese = preprocess_dataset_audio(df_chinese)

df_chinese["length"] = df_chinese["Text_interviewer_participant"].apply(len)

length_stats = df_chinese.groupby("Diagnosis")["length"].agg(["min", "max", "mean", "std"])
df_chinese = df_chinese[df_chinese.apply(filter_by_length, axis=1, stats=length_stats)]

train_cha, test_cha = train_test_split(
    df_chinese, test_size=0.2, stratify=df_chinese["Diagnosis"], random_state=42
)

train_cha.to_json(os.path.join(OUT_DIR, "train_chinese.jsonl"), orient="records", lines=True, force_ascii=False)
test_cha.to_json(os.path.join(OUT_DIR,  "test_chinese.jsonl"),  orient="records", lines=True, force_ascii=False)

print(f"Chinese: {len(df_chinese)} records after cleaning")
print(df_chinese["Diagnosis"].value_counts().to_string())
print(f"Train: {len(train_cha)}  Test: {len(test_cha)}")
