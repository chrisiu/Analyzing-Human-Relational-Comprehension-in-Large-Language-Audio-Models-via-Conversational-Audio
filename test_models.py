import os

# CACHE_DIR = INSERT YOUR CACHE DIR
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

import torch
import gc

print("MODEL LOADING TEST - 3 MODELS")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

os.makedirs(CACHE_DIR, exist_ok=True)

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

print("\nSETUP COMPLETE\n")
