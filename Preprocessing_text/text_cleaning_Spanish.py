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
    os.path.join(JSONL_DIR, "Spanish_Ivanova_output.jsonl"),
    os.path.join(JSONL_DIR, "Spanish_PerLA_output.jsonl"),
]
output_filename = "combined_jsonl_spanish_perla_Ivanova.jsonl"
combiner = JSONLCombiner(input_files, OUT_DIR, output_filename)
combiner.combine()


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


def process_transcripts(df, word_limits):
    def process_text(text, min_words, max_words):
        words = text.split()
        if len(words) < min_words:
            return None
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    processed_data = []
    for index, row in df.iterrows():
        dataset_type = row["Dataset"]
        text = row["Text_interviewer_participant"]
        if dataset_type in word_limits:
            min_words, max_words = word_limits[dataset_type]
            processed_text = process_text(text, min_words, max_words)
            if processed_text:
                row["Text_interviewer_participant"] = processed_text
                processed_data.append(row)
    return pd.DataFrame(processed_data)


df_spanish = pd.read_json(os.path.join(OUT_DIR, output_filename), lines=True)

# Map PerLA diagnosis alias
df_spanish["Diagnosis"] = df_spanish["Diagnosis"].replace("DTA", "AD")

# Drop rows with missing or Unknown diagnosis
df_spanish = df_spanish[
    df_spanish["Diagnosis"].notnull() &
    (df_spanish["Diagnosis"].str.strip() != "") &
    (df_spanish["Diagnosis"] != "Unknown")
]

df_spanish["Text_interviewer_participant"] = df_spanish["Text_interviewer_participant"].apply(preprocess_text)
df_spanish["length"] = df_spanish["Text_interviewer_participant"].apply(lambda x: len(str(x).split()))

word_limits = {
    "Ivanova": (40, 100),
    "PerLA":   (250, 1500),
}
processed_df = process_transcripts(df_spanish, word_limits)

train_df_spa, test_df_spa = train_test_split(
    processed_df, test_size=0.2, stratify=processed_df["Diagnosis"], random_state=42
)

train_df_spa.to_json(os.path.join(OUT_DIR, "train_spanish.jsonl"), orient="records", lines=True, force_ascii=False)
test_df_spa.to_json(os.path.join(OUT_DIR,  "test_spanish.jsonl"),  orient="records", lines=True, force_ascii=False)

print(f"Spanish: {len(processed_df)} records after cleaning")
print(processed_df["Diagnosis"].value_counts().to_string())
print(f"Train: {len(train_df_spa)}  Test: {len(test_df_spa)}")
