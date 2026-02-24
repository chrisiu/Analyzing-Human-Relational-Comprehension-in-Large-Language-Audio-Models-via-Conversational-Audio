import re
import types
import os


import json
import pandas as pd
from pathlib import Path
import torch
import librosa
from tqdm import tqdm
import numpy as np
import sys
import gc
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = # INSERT YOUR CACHE DIRECTORY HERE
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['TORCH_HOME'] = CACHE_DIR

INPUT_CSV = "lalm_nat_imp.csv"
OUTPUT_CSV = "lalm_nat_imp_predictions.csv"
AUDIO_DIR = ""

MAX_RETRIES = 3

DURATIONS = [None]

LABELS = ['naturalistic', 'improvised']

CLASSIFICATION_INSTRUCTION = """
CRITICAL: You must ONLY respond in English. 

SYSTEM ROLE:
You are an expert in paralinguistic analysis. Your goal is to distinguish between naturalistic human conversations and improvised role-play conversations.

TASK:
Analyze the provided audio recording of a conversation between two speakers. Your task is to classify the nature of the interaction based on the following definitions:

DEFINITIONS:
1. naturalistic: A real conversation between people who may be strangers or have an existing relationship.
2. improvised: An interaction between two actors who are role-playing a fictional scenario or persona. 

INSTRUCTIONS:
You must provide an explanation of your observations and thought-process first. Observe both the auditory and semantic/contextual characteristics of the interaction to inform your decision, then provide the final label.

RESPONSE FORMAT:
Explanation: [2-3 sentences detailing your observations and reasoning.]
Label: [naturalistic/improvised]
"""

def test_model_loading():
    print("\nMODEL LOADING TEST - 3 MODELS\n")
    
    print(f"Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\nTEST 1: AudioFlamingo3\n")
    
    try:
        from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
        
        print("Loading processor...")
        af3_processor = AutoProcessor.from_pretrained(
            "nvidia/audio-flamingo-3-hf",
            cache_dir=CACHE_DIR
        )
        print("✓ Processor loaded")
        
        print("Loading model...")
        af3_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            "nvidia/audio-flamingo-3-hf",
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ Model loaded")
        
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory Used: {mem_used:.1f} GB")
        
        del af3_processor, af3_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✓ Cleaned up")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTEST 2: Qwen2.5-Omni-7B\n")
    
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        
        print("Loading processor...")
        qwen25_processor = Qwen2_5OmniProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        print("✓ Processor loaded")
        
        print("Loading model...")
        qwen25_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ Model loaded")
        
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory Used: {mem_used:.1f} GB")
        
        del qwen25_processor, qwen25_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✓ Cleaned up")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTEST 3: R1-AQA\n")
    
    try:
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        
        print("Loading processor...")
        r1aqa_processor = AutoProcessor.from_pretrained(
            "mispeech/r1-aqa",
            cache_dir=CACHE_DIR
        )
        print("✓ Processor loaded")
        
        print("Loading model...")
        r1aqa_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "mispeech/r1-aqa",
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ Model loaded")
        
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            print(f"GPU Memory Used: {mem_used:.1f} GB")
        
        del r1aqa_processor, r1aqa_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✓ Cleaned up")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nSETUP TEST COMPLETE\n")


def load_audio_segment(audio_path, duration=None, random_segment=False):
    y, sr = librosa.load(audio_path, sr=16000, duration=None)
    
    if duration is None:
        return y, sr
    
    duration_samples = int(duration * sr)
    
    if len(y) <= duration_samples:
        return y, sr
    
    if random_segment:
        max_start = len(y) - duration_samples
        start = np.random.randint(0, max_start)
        return y[start:start + duration_samples], sr
    else:
        return y[:duration_samples], sr


def parse_response(text):
    if not text:
        return "no_output", text
    
    text = text.strip()
    
    label_match = re.search(r'Label:\s*([^\n]+)', text, re.IGNORECASE)
    explanation_match = re.search(r'Explanation:\s*(.+?)(?=Label:|$)', text, re.IGNORECASE | re.DOTALL)
    
    if label_match:
        label = label_match.group(1).strip().lower()
        
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            explanation = explanation.replace('\n', ' ').replace('\r', ' ')
        else:
            explanation = text.replace('\n', ' ').replace('\r', ' ')
        
        for valid_label in LABELS:
            if valid_label in label or label in valid_label:
                return valid_label, explanation
        
        return label, explanation
    
    text_lower = text.lower()
    for label in LABELS:
        if f" {label} " in f" {text_lower} " or text_lower.startswith(label) or text_lower.endswith(label):
            return label, text.replace('\n', ' ')
    
    for label in LABELS:
        if label in text_lower:
            return label, text.replace('\n', ' ')
    
    return text[:50], text.replace('\n', ' ')


def load_audioflamingo3():
    from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
    
    print("Loading AudioFlamingo3...")
    model_id = "nvidia/audio-flamingo-3-hf"
    
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("✓ AudioFlamingo3 loaded")
    return processor, model


def query_audioflamingo3(processor, model, audio_path, duration=None):
    try:
        random_segment = (duration == 30)
        y, sr = load_audio_segment(audio_path, duration, random_segment)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CLASSIFICATION_INSTRUCTION},
                    {"type": "audio", "audio": y, "sampling_rate": sr},
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float32:
                    inputs[k] = v.to(device=model.device, dtype=torch.float16)
                else:
                    inputs[k] = v.to(device=model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        
        decoded = processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return parse_response(decoded)
    
    except Exception as e:
        print(f"  AF3 Error: {e}")
        import traceback
        traceback.print_exc()
        return "error", str(e)


def load_qwen25omni():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    
    print("Loading Qwen2.5-Omni...")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("✓ Qwen2.5-Omni loaded")
    return processor, model


def query_qwen25omni(processor, model, audio_path, duration=None):
    try:
        random_segment = (duration == 30)
        y, sr = load_audio_segment(audio_path, duration, random_segment)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CLASSIFICATION_INSTRUCTION},
                    {"type": "audio", "audio": y, "sampling_rate": sr},
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float32:
                    inputs[k] = v.to(device=model.device, dtype=torch.float16)
                else:
                    inputs[k] = v.to(device=model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False, 
                return_audio=False
            )
        
        if isinstance(output, tuple):
            generated_ids = output[0]
        else:
            generated_ids = output
        
        decoded = processor.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()
        
        return parse_response(decoded)
    
    except Exception as e:
        print(f"  Qwen2.5-Omni Error: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return "error", str(e)


def load_r1aqa():
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    
    print("Loading R1-AQA...")
    processor = AutoProcessor.from_pretrained(
        "mispeech/r1-aqa",
        cache_dir=CACHE_DIR
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "mispeech/r1-aqa",
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ R1-AQA loaded")
    return processor, model


def query_r1aqa(processor, model, audio_path, duration=None):
    try:
        import torchaudio
        
        waveform, sampling_rate = torchaudio.load(str(audio_path))
        if sampling_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(waveform)
        audios = [waveform[0].numpy()]
        
        prompt = f"{CLASSIFICATION_INSTRUCTION}"
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        texts = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=texts, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=300)
        
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()
        
        return parse_response(response)
    
    except Exception as e:
        print(f"  R1-AQA Error: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return "error", str(e)

        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["AudioFlamingo3", "Qwen2.5-Omni", "R1-AQA", "all"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Test model loading only")
    args = parser.parse_args()
    
    if args.test:
        test_model_loading()
        return
    
    if Path(OUTPUT_CSV).exists():
        print(f"Loading existing output CSV: {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        print(f"Creating new output CSV from: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)

    if 'af3_pred' not in df.columns:
        df['af3_pred'] = None
    if 'af3_explanation' not in df.columns:
        df['af3_explanation'] = None
    if 'qwen2_5_omni_pred' not in df.columns:
        df['qwen2_5_omni_pred'] = None
    if 'qwen2_5_omni_explanation' not in df.columns:
        df['qwen2_5_omni_explanation'] = None
    if 'r1aqa_pred' not in df.columns:
        df['r1aqa_pred'] = None
    if 'r1aqa_explanation' not in df.columns:
        df['r1aqa_explanation'] = None

    if args.limit:
        df = df.head(args.limit)

    print(f"Processing {len(df)} files")
    
    models_to_run = ["AudioFlamingo3", "Qwen2.5-Omni", "R1-AQA"] if args.model == "all" else [args.model]
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Processing with {model_name}")
        print(f"{'='*60}\n")
        
        if model_name == "AudioFlamingo3":
            pred_col = 'af3_pred'
            expl_col = 'af3_explanation'
            query_func = query_audioflamingo3
        elif model_name == "Qwen2.5-Omni":
            pred_col = 'qwen2_5_omni_pred'
            expl_col = 'qwen2_5_omni_explanation'
            query_func = query_qwen25omni
        elif model_name == "R1-AQA":
            pred_col = 'r1aqa_pred'
            expl_col = 'r1aqa_explanation'
            query_func = query_r1aqa

        needs_processing = df[pred_col].isna().sum()
        print(f"Rows needing {model_name} predictions: {needs_processing}/{len(df)}")
        
        if needs_processing == 0:
            print(f"✓ All rows already have {model_name} predictions, skipping")
            continue
        
        if model_name == "AudioFlamingo3":
            processor, model = load_audioflamingo3()
        elif model_name == "Qwen2.5-Omni":
            processor, model = load_qwen25omni()
        elif model_name == "R1-AQA":
            processor, model = load_r1aqa()
        
        processed_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
            if pd.notna(df.at[idx, pred_col]):
                continue
            
            audio_filename = row['mixed_audio_filename']
            audio_path = Path(AUDIO_DIR) / audio_filename
            
            if not audio_path.exists():
                df.at[idx, pred_col] = "file_not_found"
                df.at[idx, expl_col] = f"Audio file not found: {audio_path}"
                continue
            
            pred, explanation = query_func(processor, model, str(audio_path), None)
            
            df.at[idx, pred_col] = pred
            df.at[idx, expl_col] = explanation
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                df.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
                print(f"  Saved progress: {processed_count} files processed")

        df.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
        print(f"\n✓ Completed {model_name}: {processed_count} files processed")
        
        error_rows = df[df[pred_col] == 'error']
        if len(error_rows) > 0:
            print(f"\n{'='*60}")
            print(f"RETRY PHASE: {len(error_rows)} error(s) detected")
            print(f"{'='*60}\n")
            
            for attempt in range(1, MAX_RETRIES + 1):
                error_rows = df[df[pred_col] == 'error']
                if len(error_rows) == 0:
                    print(f"✓ All errors resolved!")
                    break
                
                print(f"\n--- Retry Attempt {attempt}/{MAX_RETRIES} for {len(error_rows)} files ---\n")
                
                retry_count = 0
                for idx, row in error_rows.iterrows():
                    audio_filename = row['mixed_audio_filename']
                    audio_path = Path(AUDIO_DIR) / audio_filename
                    
                    print(f"  Retrying: {audio_filename}")
                    
                    pred, explanation = query_func(processor, model, str(audio_path), None)
                    
                    df.at[idx, pred_col] = pred
                    df.at[idx, expl_col] = explanation
                    
                    if pred != "error":
                        retry_count += 1
                        print(f"    ✓ Success: {pred}")
                    else:
                        print(f"    ✗ Still error: {explanation[:80]}...")
                    
                    time.sleep(1)
                    gc.collect()
                    torch.cuda.empty_cache()
                
                df.to_csv(OUTPUT_CSV, index=False, quoting=1, escapechar='\\')
                print(f"\n  Fixed {retry_count} errors in attempt {attempt}")
                
                if attempt < MAX_RETRIES:
                    remaining = (df[pred_col] == 'error').sum()
                    if remaining > 0:
                        print(f"  {remaining} errors remaining")
                        time.sleep(5)
            
            final_errors = (df[pred_col] == 'error').sum()
            if final_errors > 0:
                print(f"\n{final_errors} files still have errors")
        
        if 'label' in df.columns:
            correct = (df[pred_col] == df['label']).sum()
            valid = df[pred_col].notna().sum()
            error_count = (df[pred_col] == 'error').sum()
            if valid > 0:
                accuracy = correct / valid
                print(f"\n  Accuracy: {correct}/{valid} ({accuracy:.2%})")
                if error_count > 0:
                    print(f"  Errors: {error_count} files")
        
        del processor, model
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\nOutput saved to: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
