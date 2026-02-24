import csv
import shutil
import json
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS

CSV_PATH = Path("fully_merged_dataset.csv")
OUTPUT_CSV = Path("audiofiles_transcripts.csv")
OUTPUT_AUDIO_DIR = Path("./audio_files")
LOCAL_DOWNLOAD_DIR = Path("./tmp_downloads")

OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def extract_transcript(json_path):
    if not json_path or not json_path.exists():
        return ""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            segments = data.get('metadata:transcript', [])
            return " ".join([s.get('transcript', '') for s in segments if s.get('transcript')]).strip()
    except Exception:
        return ""

def mix_audio_files(audio_path1, audio_path2, output_path):
    try:
        sr1, audio1 = wavfile.read(audio_path1)
        sr2, audio2 = wavfile.read(audio_path2)
        if sr1 != sr2:
            return False

        max_len = max(len(audio1), len(audio2))
        audio1 = np.pad(audio1, (0, max_len - len(audio1)), mode='constant')
        audio2 = np.pad(audio2, (0, max_len - len(audio2)), mode='constant')

        def to_float32(x):
            if np.issubdtype(x.dtype, np.floating):
                return x.astype(np.float32)
            info = np.iinfo(x.dtype)
            return (x.astype(np.float32) / max(abs(info.min), info.max))

        mixed_f = (to_float32(audio1) + to_float32(audio2)) / 2.0
        mixed = (np.clip(mixed_f, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(output_path, sr1, mixed)
        return True
    except Exception:
        return False

def find_file(base_dir, filename):
    for p in base_dir.rglob(filename):
        return p
    return None

def process_interactions():
    df = pd.read_csv(CSV_PATH)
    
    if 'transcript_a' not in df.columns:
        df['transcript_a'] = ""
    if 'transcript_b' not in df.columns:
        df['transcript_b'] = ""

    repaired = 0
    for idx, row in df.iterrows():
        mixed_name = row["mixed_audio_filename"]
        file_id_a, file_id_b = row["file_id_a"], row["file_id_b"]
        out_path = OUTPUT_AUDIO_DIR / mixed_name

        config = DatasetConfig(
            label=row["label"],
            split=row.get("split", "train"),
            preferred_vendors_only=True,
            num_workers=2,
            local_dir=LOCAL_DOWNLOAD_DIR
        )
        fs = SeamlessInteractionFS(config=config)

        try:
            fs.download_batch_from_s3([file_id_a, file_id_b])
        except Exception:
            pass

        audio_a = find_file(LOCAL_DOWNLOAD_DIR, f"{file_id_a}.wav")
        audio_b = find_file(LOCAL_DOWNLOAD_DIR, f"{file_id_b}.wav")
        json_a = find_file(LOCAL_DOWNLOAD_DIR, f"{file_id_a}.json")
        json_b = find_file(LOCAL_DOWNLOAD_DIR, f"{file_id_b}.json")

        df.at[idx, 'transcript_a'] = extract_transcript(json_a)
        df.at[idx, 'transcript_b'] = extract_transcript(json_b)

        if audio_a and audio_b and not out_path.exists():
            if mix_audio_files(audio_a, audio_b, out_path):
                repaired += 1
        
        shutil.rmtree(LOCAL_DOWNLOAD_DIR, ignore_errors=True)
        LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Process complete. {repaired} files mixed. Saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    process_interactions()
