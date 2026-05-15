import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from processing.phase1.common import (
    DATA_ROOT,
    PHASE1_ROOT,
    TABLES_PHASE1_RESULT_TABLES,
    TABLES_PHASE1_SUMMARIES,
    clean_text,
    get_analysis_text,
    load_cleaned_full_corpus,
    make_logger,
    normalize_language,
    stable_id,
)


MANIFEST_PATH = PHASE1_ROOT / "phase1_manifest.jsonl"


TASK_MARKER_FAMILIES = {
    "activity": ("PD_CTP", "cookie_theft"),
    "cookie": ("PD_CTP", "cookie_theft"),
    "cookie theft": ("PD_CTP", "cookie_theft"),
    "pict_descr": ("PD_CTP", "picture_description"),
    "picture_description": ("PD_CTP", "picture_description"),
    "cat": ("PICTURE_STORY", "cat_rescue"),
    "rockwell": ("PICTURE_STORY", "going_and_coming"),
    "window": ("PICTURE_STORY", "window_story"),
    "umbrella": ("PICTURE_STORY", "umbrella_story"),
    "going and coming": ("PICTURE_STORY", "going_and_coming"),
    "cinderella": ("STORY_NARRATIVE", "cinderella"),
    "cinderella_intro": ("STORY_NARRATIVE", "cinderella"),
    "cinderella_pictures": ("STORY_NARRATIVE", "cinderella"),
    "red_riding_hood": ("STORY_NARRATIVE", "red_riding_hood"),
    "story": ("STORY_NARRATIVE", "story"),
    "sandwich": ("PROCEDURAL", "sandwich"),
    "sandwich_favorite": ("PROCEDURAL", "favorite_sandwich"),
    "favorite_sandwich": ("PROCEDURAL", "favorite_sandwich"),
    "tea": ("PROCEDURAL", "tea"),
    "important_event": ("PERSONAL_NARRATIVE", "important_event"),
    "routine": ("PERSONAL_NARRATIVE", "routine"),
    "hometown": ("PERSONAL_NARRATIVE", "hometown"),
    "conversation": ("CONVERSATION", "conversation"),
    "con": ("CONVERSATION", "conversation"),
    "colloquial conversation": ("CONVERSATION", "conversation"),
    "voice_commands": ("COMMAND", "voice_commands"),
    "reading": ("READING", "reading"),
    "fluency": ("FLUENCY", "fluency"),
    "verbal_fluency": ("FLUENCY", "fluency"),
    "repetition": ("REPETITION", "repetition"),
    "bnt": ("OTHER", "bnt"),
}


MIXED_PROTOCOL_DATASETS = {"Baycrest", "Delaware"}


def build_path_index(root: Path, extensions: set[str]) -> dict[str, list[str]]:
    index = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext not in extensions:
                continue
            stem = Path(filename).stem
            index[stem].append(str(Path(dirpath) / filename))
    return index


def infer_dataset_name(row: pd.Series) -> str:
    dataset = str(row.get("Dataset", "")).strip()
    file_id = str(row.get("File_ID", "")).strip()
    source_lang = str(row.get("source_language_name", ""))

    if dataset and dataset != "Unknown":
        return dataset

    if file_id.startswith("taukdial-"):
        return "TAUKADIAL"
    if file_id.startswith("madrs"):
        return "ADReSS-M"
    if file_id.startswith("Patient"):
        if (DATA_ROOT / "Greek" / "DS5" / file_id).exists():
            return "DS5"
        if (DATA_ROOT / "Greek" / "DS7" / file_id).exists():
            return "DS7"
    if any(token in file_id.lower() for token in ("audiorecday", "audiorec_day", "controlday", "patientsday")):
        return "DS3"
    if "ncmmsc" in file_id.lower():
        return "NCMMSC2021_AD"
    if source_lang == "greek":
        return "Greek_ASR_Unknown"
    if source_lang == "english":
        return "English_Unknown"
    if source_lang == "chinese":
        return "Chinese_Unknown"
    return dataset or "Unknown"


def infer_participant_id(dataset_name: str, row: pd.Series) -> tuple[str, str]:
    pid = clean_text(row.get("PID", ""))
    file_id = clean_text(row.get("File_ID", ""))

    if dataset_name == "TAUKADIAL":
        parts = file_id.split("-")
        if len(parts) >= 3:
            return f"taukadial_{parts[1]}", "file_id_middle_segment"

    if dataset_name in {"DS5", "DS7"} and file_id.startswith("Patient"):
        return file_id, "file_id_patient"

    if dataset_name == "ADReSS-M" and file_id.startswith("madrs"):
        return file_id, "file_id_madrs"

    if dataset_name == "DS3":
        match = re.search(r"patient\s+(\d+)", file_id.lower())
        if match:
            return f"ds3_patient_{match.group(1)}", "path_patient_segment"
        match = re.search(r"control\s+(\d+)", file_id.lower())
        if match:
            return f"ds3_control_{match.group(1)}", "path_control_segment"

    if dataset_name == "NCMMSC2021_AD":
        stem = Path(file_id).stem
        parts = stem.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:3]), "filename_subject_segment"

    if dataset_name == "Predictive_Chinese_challenge_Chinese_2019" and pid and pid != "Unknown":
        return pid, "pid"

    if dataset_name in {"Pitt", "Delaware", "Ivanova"} and file_id and "-" in file_id:
        return file_id.rsplit("-", 1)[0], "file_id_prefix"

    if dataset_name == "Kempler" and file_id:
        match = re.match(r"(d\d+)", file_id.lower())
        if match:
            return match.group(1), "file_id_prefix"

    if pid and pid != "Unknown":
        return pid, "pid"
    if file_id:
        return file_id, "file_id"
    return stable_id(dataset_name, str(row.name)), "synthetic_hash"


def map_diagnosis(raw: str) -> tuple[str, int | None]:
    raw = str(raw).strip()
    mapped = raw
    if raw in {"AD", "ProbableAD", "PossibleAD", "Dementia"}:
        mapped = "AD"
        return mapped, 1
    if raw == "HC":
        return "HC", 0
    if raw == "MCI":
        return "MCI", None
    return mapped or "Unknown", None


def infer_label_metadata(dataset_name: str) -> tuple[str, str]:
    if dataset_name == "WLS":
        return "spreadsheet_rule_join", "weak_rule_based"
    if dataset_name in {"DS3", "DS5", "DS7", "Kempler", "VAS"}:
        return "metadata_join", "derived_from_metadata"
    return "dataset_provided_or_extracted", "dataset_provided"


def resolve_audio_path(dataset_name: str, file_id: str, audio_index: dict[str, list[str]]) -> str:
    if dataset_name == "ADReSS-M":
        target = DATA_ROOT / "ADReSS-M" / "ADReSS-M-test-gr" / "test-gr" / f"{file_id}.wav"
        return str(target) if target.exists() else ""

    if dataset_name in {"DS5", "DS7"}:
        folder = DATA_ROOT / "Greek" / dataset_name / file_id
        if folder.exists():
            wavs = sorted(folder.glob("*.wav"))
            if wavs:
                return str(wavs[0])

    if dataset_name == "DS3":
        target = DATA_ROOT / "Greek" / "DS3" / f"{file_id}.wav"
        if target.exists():
            return str(target)

    if dataset_name == "TAUKADIAL":
        train = DATA_ROOT / "TAUKADIAL" / "TAUKADIAL-24-train" / "TAUKADIAL-24" / "train" / f"{file_id}.wav"
        test = DATA_ROOT / "TAUKADIAL" / "TAUKADIAL-24-test" / "TAUKADIAL-24" / "test" / f"{file_id}.wav"
        if train.exists():
            return str(train)
        if test.exists():
            return str(test)

    if dataset_name == "NCMMSC2021_AD":
        target = DATA_ROOT / "NCMMSC2021_AD" / "AD_dataset_long" / f"{file_id}.wav"
        if target.exists():
            return str(target)

    stem = Path(file_id).stem
    candidates = audio_index.get(stem, [])
    if len(candidates) == 1:
        return candidates[0]

    normalized = file_id.replace("\\", "/")
    if "/" in normalized:
        suffix = f"{normalized}.wav"
        matches = [candidate for candidate in candidates if candidate.replace("\\", "/").endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
    return ""


def resolve_transcript_path(dataset_name: str, row: pd.Series, text_index: dict[str, list[str]]) -> str:
    file_id = str(row.get("File_ID", "")).strip()
    pid = str(row.get("PID", "")).strip()

    if dataset_name == "Predictive_Chinese_challenge_Chinese_2019" and pid and pid != "Unknown":
        target = DATA_ROOT / "Chinese" / "iFlytek" / f"{pid}.tsv"
        return str(target) if target.exists() else ""

    stem = file_id if file_id and file_id != "Unknown" else pid
    if not stem:
        return ""

    candidates = text_index.get(Path(stem).stem, [])
    if not candidates:
        return ""

    if dataset_name == "Pitt":
        dx = str(row.get("Diagnosis", "")).strip()
        subgroup = "Control" if dx == "HC" else "Dementia"
        target_root = str(DATA_ROOT / "English" / "Pitt" / subgroup / "cookie")
        matches = [path for path in candidates if target_root in path]
        if len(matches) == 1:
            return matches[0]

    dataset_hint = {
        "Ivanova": str(DATA_ROOT / "Spanish" / "Ivanova"),
        "PerLA": str(DATA_ROOT / "Spanish" / "PerLA"),
        "Baycrest": str(DATA_ROOT / "English" / "Baycrest"),
        "Delaware": str(DATA_ROOT / "English" / "Delaware"),
        "Kempler": str(DATA_ROOT / "English" / "Kempler"),
        "Lu": str(DATA_ROOT / "English" / "Lu"),
        "WLS": str(DATA_ROOT / "English" / "WLS"),
        "VAS": str(DATA_ROOT / "English" / "VAS"),
    }.get(dataset_name)

    if dataset_hint:
        matches = [path for path in candidates if dataset_hint in path]
        if len(matches) == 1:
            return matches[0]

    return candidates[0] if len(candidates) == 1 else ""


def normalize_marker_text(raw_marker: str) -> str:
    marker = clean_text(raw_marker).lower()
    marker = marker.replace("favorite_sandiwch", "favorite_sandwich")
    marker = marker.replace("sandiwch_faorite", "sandwich_favorite")
    marker = marker.replace("sandiwch", "sandwich")
    marker = marker.replace("sanwich", "sandwich")
    marker = marker.replace("sanswich", "sandwich")
    marker = marker.replace("cinderlla", "cinderella")
    marker = marker.replace("cionderella", "cinderella")
    marker = marker.replace("cinderella_inro", "cinderella_intro")
    marker = marker.replace("cinderella_intro", "cinderella_intro")
    marker = marker.replace("cinderella intro", "cinderella_intro")
    marker = marker.replace("cookie theft", "cookie")
    return marker.strip()


def extract_transcript_markers(transcript_path: str) -> list[str]:
    if not transcript_path:
        return []
    path = Path(transcript_path)
    if not path.exists():
        return []

    markers = []
    if path.suffix.lower() == ".cha":
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            for prefix in ("@G:", "@Situation:", "@Activities:", "@Bg:"):
                if line.startswith(prefix):
                    markers.append(line.split(":", 1)[1].strip())
                    break
    elif path.suffix.lower() == ".tsv":
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in ("图片", "看这张图片", "看图", "告诉我你看到")):
            markers.append("cookie")
    return markers


def categorize_markers(markers: list[str]) -> tuple[list[str], list[str]]:
    components = []
    prompt_ids = []
    for marker in markers:
        normalized = normalize_marker_text(marker)
        family = TASK_MARKER_FAMILIES.get(normalized)
        if family is None:
            for key, value in TASK_MARKER_FAMILIES.items():
                if key and key in normalized:
                    family = value
                    break
        if family is None:
            continue
        components.append(family[0])
        prompt_ids.append(family[1])
    return sorted(set(components)), sorted(set(prompt_ids))


def infer_ds3_task(file_id: str) -> tuple[str, str, str, str]:
    normalized = file_id.replace("\\", "/").lower()
    if "/test1/" in normalized or normalized.endswith("/test1"):
        return "PD_CTP", "ds3_path_rule", "medium", "ds3_test1_picture_description"
    if "/test2/" in normalized or normalized.endswith("/test2"):
        return "PICTURE_RECALL", "ds3_path_rule", "medium", "ds3_test2_picture_recall"
    if "/test3/" in normalized or normalized.endswith("/test3"):
        return "REPETITION", "ds3_path_rule", "medium", "ds3_test3_repetition"
    if "/test4/" in normalized or normalized.endswith("/test4"):
        return "MOTOR_SPEECH", "ds3_path_rule", "medium", "ds3_test4_motor_speech"
    return "OTHER", "ds3_path_rule", "low", ""


def infer_task_fields(dataset_name: str, row: pd.Series, transcript_path: str) -> dict[str, str | int]:
    file_id = str(row.get("File_ID", "")).strip()
    transcript_markers = extract_transcript_markers(transcript_path)
    task_components, prompt_ids = categorize_markers(transcript_markers)

    result = {
        "task_type": "OTHER",
        "task_type_source": "fallback_rule",
        "task_type_confidence": "low",
        "prompt_id": "",
        "task_components": "|".join(task_components),
        "task_component_count": len(task_components),
        "task_markers_raw": "|".join(clean_text(marker) for marker in transcript_markers),
        "task_is_mixed_protocol": 0,
    }

    if dataset_name == "Pitt":
        result.update({
            "task_type": "PD_CTP",
            "task_type_source": "paper_rule",
            "task_type_confidence": "high",
            "prompt_id": "cookie_theft",
            "task_components": "PD_CTP",
            "task_component_count": 1,
        })
        return result

    if dataset_name == "Lu":
        result.update({
            "task_type": "PD_CTP",
            "task_type_source": "talkbank_corpus_page",
            "task_type_confidence": "high",
            "prompt_id": "cookie_theft",
            "task_components": "PD_CTP",
            "task_component_count": 1,
        })
        return result

    if dataset_name == "WLS":
        result.update({
            "task_type": "PD_CTP",
            "task_type_source": "transcript_prompt_rule",
            "task_type_confidence": "high",
            "prompt_id": "cookie_theft",
            "task_components": "PD_CTP",
            "task_component_count": 1,
        })
        return result

    if dataset_name == "VAS":
        result.update({
            "task_type": "COMMAND",
            "task_type_source": "transcript_prompt_rule",
            "task_type_confidence": "high",
            "prompt_id": "voice_commands",
            "task_components": "COMMAND",
            "task_component_count": 1,
        })
        return result

    if dataset_name == "PerLA":
        result.update({
            "task_type": "CONVERSATION",
            "task_type_source": "transcript_marker_rule",
            "task_type_confidence": "high",
            "prompt_id": "conversation",
            "task_components": "CONVERSATION",
            "task_component_count": 1,
        })
        return result

    if dataset_name == "Ivanova":
        result.update({
            "task_type": "READING",
            "task_type_source": "paper_rule",
            "task_type_confidence": "high",
            "prompt_id": "reading",
            "task_components": "READING",
            "task_component_count": 1,
        })
        return result

    if dataset_name == "Predictive_Chinese_challenge_Chinese_2019":
        result.update({
            "task_type": "PD_CTP",
            "task_type_source": "transcript_prompt_rule",
            "task_type_confidence": "high",
            "prompt_id": "cookie_theft",
            "task_components": "PD_CTP",
            "task_component_count": 1,
        })
        return result

    if dataset_name in {"DS5", "DS7", "ADReSS-M", "TAUKADIAL"}:
        prompt_id = "pict_descr" if dataset_name in {"DS5", "DS7"} else "picture_description"
        result.update({
            "task_type": "PD_CTP",
            "task_type_source": "pipeline_rule",
            "task_type_confidence": "high",
            "prompt_id": prompt_id,
            "task_components": "PD_CTP",
            "task_component_count": 1,
        })
        return result

    if dataset_name == "DS3":
        task_type, source, confidence, prompt_id = infer_ds3_task(file_id)
        result.update({
            "task_type": task_type,
            "task_type_source": source,
            "task_type_confidence": confidence,
            "prompt_id": prompt_id,
            "task_components": task_type,
            "task_component_count": 1,
        })
        return result

    if dataset_name == "NCMMSC2021_AD":
        result.update({
            "task_type": "OTHER",
            "task_type_source": "lossy_long_audio_packaging",
            "task_type_confidence": "low",
            "prompt_id": "",
        })
        return result

    if dataset_name == "Kempler" and task_components:
        if len(task_components) == 1:
            result.update({
                "task_type": task_components[0],
                "task_type_source": "transcript_marker_rule",
                "task_type_confidence": "high",
                "prompt_id": prompt_ids[0] if prompt_ids else "",
            })
        else:
            result.update({
                "task_type": "MIXED_PROTOCOL",
                "task_type_source": "transcript_marker_rule",
                "task_type_confidence": "high",
                "prompt_id": "mixed_protocol",
                "task_is_mixed_protocol": 1,
            })
        return result

    if dataset_name in MIXED_PROTOCOL_DATASETS and task_components:
        if len(task_components) == 1:
            result.update({
                "task_type": task_components[0],
                "task_type_source": "transcript_marker_rule",
                "task_type_confidence": "high",
                "prompt_id": prompt_ids[0] if prompt_ids else "",
            })
        else:
            result.update({
                "task_type": "MIXED_PROTOCOL",
                "task_type_source": "transcript_marker_rule",
                "task_type_confidence": "high",
                "prompt_id": "mixed_protocol",
                "task_is_mixed_protocol": 1,
            })
        return result

    if task_components:
        result.update({
            "task_type": task_components[0] if len(task_components) == 1 else "MIXED_PROTOCOL",
            "task_type_source": "transcript_marker_rule",
            "task_type_confidence": "medium",
            "prompt_id": prompt_ids[0] if len(prompt_ids) == 1 else "mixed_protocol",
            "task_is_mixed_protocol": int(len(task_components) > 1),
        })
        return result

    return result


def build_manifest() -> pd.DataFrame:
    log = make_logger("phase1_build_manifest")
    log("Loading cleaned corpora")
    corpus = load_cleaned_full_corpus().reset_index(drop=True)

    log("Indexing raw audio and transcript files")
    audio_index = build_path_index(DATA_ROOT, {".wav", ".mp3"})
    text_index = build_path_index(DATA_ROOT, {".cha", ".tsv"})

    rows = []
    for idx, row in corpus.iterrows():
        dataset_name = infer_dataset_name(row)
        language = normalize_language(row.get("Languages", ""), row.get("source_language_name", ""))
        participant_id, participant_id_source = infer_participant_id(dataset_name, row)
        diagnosis_raw = str(row.get("Diagnosis", "")).strip()
        diagnosis_mapped, binary_label = map_diagnosis(diagnosis_raw)
        label_source, label_quality = infer_label_metadata(dataset_name)
        transcript_path = resolve_transcript_path(dataset_name, row, text_index)
        audio_path = resolve_audio_path(dataset_name, str(row.get("File_ID", "")).strip(), audio_index)
        task_fields = infer_task_fields(dataset_name, row, transcript_path)

        text_participant = clean_text(row.get("Text_participant", ""))
        text_combined = clean_text(row.get("Text_interviewer_participant", ""))
        analysis_text = get_analysis_text(row)

        sample_id = stable_id(dataset_name, str(row.get("File_ID", "")), diagnosis_raw, language, str(idx))
        group_id = participant_id

        manifest_row = {
            "sample_id": sample_id,
            "group_id": group_id,
            "participant_id": participant_id,
            "participant_id_source": participant_id_source,
            "dataset_name": dataset_name,
            "language": language,
            "task_type": task_fields["task_type"],
            "task_type_source": task_fields["task_type_source"],
            "task_type_confidence": task_fields["task_type_confidence"],
            "prompt_id": task_fields["prompt_id"],
            "task_components": task_fields["task_components"],
            "task_component_count": task_fields["task_component_count"],
            "task_markers_raw": task_fields["task_markers_raw"],
            "task_is_mixed_protocol": task_fields["task_is_mixed_protocol"],
            "diagnosis_raw": diagnosis_raw,
            "diagnosis_mapped": diagnosis_mapped,
            "binary_label": binary_label,
            "label_source": label_source,
            "label_quality": label_quality,
            "legacy_split": row.get("legacy_split", ""),
            "raw_language": str(row.get("Languages", "")),
            "source_language_name": str(row.get("source_language_name", "")),
            "source_dataset_field": str(row.get("Dataset", "")),
            "file_id": str(row.get("File_ID", "")),
            "pid_raw": str(row.get("PID", "")),
            "transcript_path": transcript_path,
            "audio_path": audio_path,
            "age": row.get("Age", ""),
            "sex": row.get("Gender", ""),
            "education": row.get("Education", ""),
            "mmse": row.get("MMSE", ""),
            "moca": row.get("Moca", ""),
            "has_text": bool(analysis_text),
            "has_audio": bool(audio_path),
            "has_participant_only_text": bool(text_participant),
            "analysis_text": analysis_text,
            "text_participant": text_participant,
            "text_combined": text_combined,
            "text_length_chars": len(analysis_text),
        }
        rows.append(manifest_row)

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(["language", "dataset_name", "participant_id", "sample_id"]).reset_index(drop=True)
    return manifest


def write_manifest_outputs(manifest: pd.DataFrame) -> None:
    manifest.to_json(MANIFEST_PATH, orient="records", lines=True, force_ascii=False)

    audit_lang = (
        manifest.groupby(["language", "task_type", "diagnosis_mapped"])
        .size()
        .reset_index(name="count")
        .sort_values(["language", "task_type", "diagnosis_mapped"])
    )
    audit_lang.to_csv(TABLES_PHASE1_RESULT_TABLES / "manifest_counts_language_task_diagnosis.csv", index=False)

    audit_dataset = (
        manifest.groupby(["dataset_name", "task_type", "diagnosis_mapped"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset_name", "task_type", "diagnosis_mapped"])
    )
    audit_dataset.to_csv(TABLES_PHASE1_RESULT_TABLES / "manifest_counts_dataset_task_diagnosis.csv", index=False)

    mixed_protocol = (
        manifest.groupby(["dataset_name", "task_is_mixed_protocol"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset_name", "task_is_mixed_protocol"])
    )
    mixed_protocol.to_csv(TABLES_PHASE1_RESULT_TABLES / "manifest_mixed_protocol_counts.csv", index=False)

    availability = (
        manifest.groupby(["language", "dataset_name"])[["has_text", "has_audio", "has_participant_only_text"]]
        .mean()
        .reset_index()
    )
    availability.to_csv(TABLES_PHASE1_RESULT_TABLES / "manifest_feature_availability.csv", index=False)

    summary = {
        "num_rows": int(len(manifest)),
        "num_groups": int(manifest["group_id"].nunique()),
        "languages": manifest["language"].value_counts().to_dict(),
        "datasets": manifest["dataset_name"].value_counts().to_dict(),
        "task_types": manifest["task_type"].value_counts().to_dict(),
        "diagnosis_mapped": manifest["diagnosis_mapped"].value_counts().to_dict(),
        "mixed_protocol_rows": int(manifest["task_is_mixed_protocol"].sum()),
    }
    with (TABLES_PHASE1_SUMMARIES / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    logger = make_logger("phase1_build_manifest")
    logger("Starting phase 1 manifest build")
    manifest_df = build_manifest()
    logger(f"Built manifest with {len(manifest_df)} rows")
    write_manifest_outputs(manifest_df)
    logger(f"Wrote manifest to {MANIFEST_PATH}")
