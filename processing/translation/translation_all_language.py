import argparse
import json
import os
import sys
import time
from datetime import datetime

import openai


parser = argparse.ArgumentParser()
parser.add_argument("--directory_to_input_data", required=True)
parser.add_argument("--directory_to_output_translated", required=True)
parser.add_argument("--source_language", required=True)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--model", default="gpt-4o")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()


def load_dotenv() -> None:
    """Load simple KEY=VALUE entries from the repo-root .env if present."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to the repo-root .env file or set it in the environment.")
if api_key.startswith("="):
    raise RuntimeError("OPENAI_API_KEY begins with '='. Fix the .env entry to look like OPENAI_API_KEY=sk-...")

client = openai.OpenAI(api_key=api_key)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def record_signature(record: dict) -> tuple[str, str, str, str]:
    return (
        str(record.get("File_ID", "")),
        str(record.get("Dataset", "")),
        str(record.get("Diagnosis", "")),
        str(record.get("Text_interviewer_participant", "")),
    )


def make_logger(log_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log(message: str) -> None:
        line = f"[{now()}] {message}"
        safe_line = line.encode("ascii", "backslashreplace").decode("ascii")
        try:
            print(safe_line, flush=True)
        except OSError:
            pass
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def translate_text(text: str, source_lang: str, model: str, target_lang: str = "English", max_chars: int = 2048) -> str:
    prompt_template = f"Translate the following {source_lang} text to {target_lang}:\n\n{{}}\n\nTranslation:"
    translated_chunks = []
    text_chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [""]

    for chunk_idx, chunk in enumerate(text_chunks, start=1):
        prompt = prompt_template.format(chunk)
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a careful translation assistant. Preserve meaning and obvious speaker structure."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                translated_chunks.append(response.choices[0].message.content.strip())
                break
            except openai.RateLimitError:
                time.sleep(2 ** attempt)
            except openai.APIError:
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)

    return " ".join(part for part in translated_chunks if part).strip()


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_prefix(output_path: str, input_records: list[dict], log, overwrite: bool) -> list[dict]:
    if not os.path.exists(output_path):
        return []

    existing = load_jsonl(output_path)
    if not existing:
        return []

    if overwrite:
        log(f"Overwrite requested; ignoring existing output with {len(existing)} rows.")
        return []

    if len(existing) > len(input_records):
        raise RuntimeError(
            f"Existing output has {len(existing)} rows but input only has {len(input_records)}. "
            "Delete the output file or rerun with --overwrite."
        )

    for idx, existing_record in enumerate(existing):
        if record_signature(existing_record) != record_signature(input_records[idx]):
            raise RuntimeError(
                "Existing output does not match the current input prefix at row "
                f"{idx + 1}. Delete the output file or rerun with --overwrite."
            )
        if "translated" not in existing_record or not str(existing_record["translated"]).strip():
            raise RuntimeError(
                f"Existing output row {idx + 1} is missing translated text. "
                "Delete the output file or rerun with --overwrite."
            )

    log(f"Resuming from existing output: {len(existing)}/{len(input_records)} rows already translated.")
    return existing


def write_record(output_handle, record: dict) -> None:
    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_handle.flush()


def translate_dataset(input_path: str, output_path: str, source_lang: str, model: str, overwrite: bool) -> None:
    log_path = output_path + ".log"
    log = make_logger(log_path)

    if args.workers != 1:
        log(f"Ignoring workers={args.workers}; translator now runs sequentially for resumable writes.")

    log(f"Input: {input_path}")
    log(f"Output: {output_path}")
    log(f"Model: {model}")

    input_records = load_jsonl(input_path)
    total = len(input_records)
    log(f"Loaded {total} input rows.")

    existing = load_existing_prefix(output_path, input_records, log, overwrite)
    translated_count = len(existing)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mode = "w" if overwrite or translated_count == 0 else "a"

    if overwrite and os.path.exists(output_path):
        log("Truncating existing output file before restart.")

    with open(output_path, mode, encoding="utf-8") as output_handle:
        if overwrite:
            output_handle.truncate(0)

        for idx in range(translated_count, total):
            record = dict(input_records[idx])
            preview = str(record.get("Text_interviewer_participant", ""))[:80].replace("\n", " ")
            file_id = str(record.get("File_ID", "Unknown"))
            log(f"[{idx + 1}/{total}] Translating File_ID={file_id} Preview={preview!r}")
            started = time.time()
            try:
                record["translated"] = translate_text(
                    str(record.get("Text_interviewer_participant", "")),
                    source_lang=source_lang,
                    model=model,
                )
            except Exception as exc:
                log(f"[{idx + 1}/{total}] FAILED: {exc.__class__.__name__}: {exc}")
                raise

            write_record(output_handle, record)
            elapsed = time.time() - started
            translated_len = len(str(record.get("translated", "")))
            log(f"[{idx + 1}/{total}] Saved in {elapsed:.1f}s translated_chars={translated_len}")

    log(f"Completed translation: {total}/{total} rows written to {output_path}")


if __name__ == "__main__":
    try:
        translate_dataset(
            input_path=args.directory_to_input_data,
            output_path=args.directory_to_output_translated,
            source_lang=args.source_language,
            model=args.model,
            overwrite=args.overwrite,
        )
    except KeyboardInterrupt:
        print(f"[{now()}] Interrupted by user.", flush=True)
        sys.exit(130)
