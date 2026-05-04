import openai
import json
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument('--directory_to_input_data', required=True)
parser.add_argument('--directory_to_output_translated', required=True)
parser.add_argument('--source_language', required=True)
parser.add_argument('--workers', type=int, default=3)
args_slurm = parser.parse_args()

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))


def translate_text(text, source_lang, target_lang="English", max_tokens=2048):
    prompt_template = f"Translate the following {source_lang} text to {target_lang}:\n\n{{}}\n\nTranslation:"
    translated_text = ""
    text_chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    for chunk in text_chunks:
        prompt = prompt_template.format(chunk)
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful translator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                translated_text += response.choices[0].message.content.strip() + " "
                break
            except openai.RateLimitError:
                wait = 2 ** attempt
                time.sleep(wait)
            except openai.APIError as e:
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)
    return translated_text.strip()


def translate_record(args):
    idx, data, source_lang = args
    text = data.get("Text_interviewer_participant", "")
    translated = translate_text(text, source_lang)
    data["translated"] = translated
    return idx, data


def translate_dataset(input_path, output_path, source_lang, workers=3):
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f]

    total = len(records)
    results = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(translate_record, (i, rec, source_lang)): i
            for i, rec in enumerate(records)
        }
        for future in as_completed(futures):
            idx, data = future.result()
            results[idx] = data
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  [{completed}/{total}] done", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"Saved {total} records → {output_path}", flush=True)


translate_dataset(
    input_path=args_slurm.directory_to_input_data,
    output_path=args_slurm.directory_to_output_translated,
    source_lang=args_slurm.source_language,
    workers=args_slurm.workers,
)
