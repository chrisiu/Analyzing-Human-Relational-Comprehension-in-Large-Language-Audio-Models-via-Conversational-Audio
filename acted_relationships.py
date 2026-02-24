import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_CSV =  "fully_merged_daset.csv"
OUTPUT_CSV = "fully_merged_daset.csv"

MODEL = "gpt-4o-mini"
BATCH_SIZE = 10
SLEEP_BETWEEN_CALLS = 0.2
batch_count = 0

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You extract the core social role from conversational prompt text.

Your task: Find the phrase after "As the" and extract ONLY the fundamental relationship type.

CRITICAL RULES:
1. Remove ALL descriptive phrases (e.g., "pushing for X", "seeking Y", "being flirted with")
2. Keep only the base relationship noun (e.g., "partner", "employee", "customer")
3. For compound roles, keep 2-3 words maximum (e.g., "help desk representative", "career counselor")
4. If the role is genuinely vague (like "person"), check context for the actual relationship:
   - "person doing the flirting" → "flirter" 
5. Output lowercase, 1-3 words maximum
6. Output ONLY the role text, nothing else

Examples:
- "As the partner pushing for efficiency" → "partner"
- "As the employee seeking a raise" → "employee"  
- "As the help desk representative handling complaints" → "help desk representative"
- "As the client negotiating with an artist" → "client"
"""


def build_batch_prompt(texts):
    numbered = "\n".join(f"{i+1}. {t[:200]}..." if len(t) > 200 else f"{i+1}. {t}" 
                         for i, t in enumerate(texts))
    return f"""Extract the core social role from each prompt text below.

Rules:
- Find "As the [role]" in each text
- Extract ONLY the base relationship type
- Strip ALL descriptive phrases and clauses
- Keep 1-3 words maximum
- For vague terms like "person", try to infer the actual role from context

Texts:
{numbered}

Output valid JSON array with extracted roles in order:
["role1", "role2", ...]"""


def call_gpt_batch(texts):
    global batch_count
    batch_count += 1
    print(f"Processing batch {batch_count} with {len(texts)} items...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(texts)},
        ],
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()

    start = content.find("[")
    end = content.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {content}")

    content = content[start:end + 1]

    return json.loads(content)


df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
df = df.reset_index(drop=True)

results_a = [""] * len(df)
results_b = [""] * len(df)

prompt_cache = {}

batch_texts = []
batch_targets = []

def flush_batch():
    global batch_texts, batch_targets

    outputs = call_gpt_batch(batch_texts)

    for (row_idx, side, prompt_text), value in zip(batch_targets, outputs):
        value = value.strip().lower()
        
        prompt_cache[prompt_text] = value

        if side == "a":
            results_a[row_idx] = value
        else:
            results_b[row_idx] = value

    batch_texts = []
    batch_targets = []
    time.sleep(SLEEP_BETWEEN_CALLS)


for idx, row in df.iterrows():
    if row["label"] != "improvised":
        continue

    for side in ["a", "b"]:
        col = f"participant_{side}_prompt_text"
        prompt = row[col]

        if not isinstance(prompt, str) or not prompt.strip():
            continue

        if prompt in prompt_cache:
            cached_value = prompt_cache[prompt]
            if side == "a":
                results_a[idx] = cached_value
            else:
                results_b[idx] = cached_value
            continue

        batch_texts.append(prompt)
        batch_targets.append((idx, side, prompt))

        if len(batch_texts) >= BATCH_SIZE:
            flush_batch()

if batch_texts:
    flush_batch()

df["acted_relationship_a"] = results_a
df["acted_relationship_b"] = results_b

df.to_csv(OUTPUT_CSV, index=False)

unique_prompts_a = df[df["label"] == "improvised"]["participant_a_prompt_text"].nunique()
unique_prompts_b = df[df["label"] == "improvised"]["participant_b_prompt_text"].nunique()
total_rows = len(df[df["label"] == "improvised"])

print(f"\nDone - {OUTPUT_CSV}")
