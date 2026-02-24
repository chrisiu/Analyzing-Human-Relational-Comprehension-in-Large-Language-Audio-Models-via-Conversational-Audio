import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import f1_score, classification_report, confusion_matrix

load_dotenv()

INPUT_CSV = "audiofiles_transcripts_merged.csv"
OUTPUT_CSV = "audiofiles_transcripts_gpt5_eval.csv"
LOG_FILE = "gpt5_predictions_log.txt"

MODEL = "gpt-5-mini"
BATCH_SIZE = 1

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

VALID_LABELS = [
    "stranger",
    "friends",
    "family-members",
    "coworkers",
    "dating/spouse/romantic_partner",
    "classmates",
    "roommates",
    "neighbors",
]

RELATIONSHIP_PROMPT = """
Determine the CURRENT relationship between Speaker A and Speaker B based only on their written dialogue. Only classify the relationship between the two speakers.

Note that the relationship typically will not be the topic of conversation, and may not even be explicitly mentioned. Instead, you must infer the relationship based on the content and tone of the conversation.

Possible relationship types:

- stranger: two people with no prior relationship meeting or speaking for the first time
- friends: two people with an established personal friendship who know each other outside of work, school, or family
- family-members: two people who are related (e.g., parent-child, siblings, cousins)
- coworkers: two people who work together in a professional/workplace setting
- dating/spouse/romantic_partner: two people in an ongoing romantic relationship (e.g., dating, married)
- classmates: two students who attend the same class or school
- roommates: two people who live together and share a living space
- neighbors: two people who live in neighboring houses or residences

You may NOT respond with ANY LABEL OTHER THAN THE ONES LISTED ABOVE. If you are unsure, make your best guess based on the conversation, but you MUST choose one of the above labels.
"""

def classify_batch(rows):
    prompt = RELATIONSHIP_PROMPT + "\n\nConversations:\n"
    for i, row in enumerate(rows, 1):
        prompt += f"\n--- Conversation {i} ---\n"
        prompt += f"Speaker A:\n{str(row['transcript_a'])[:2000]}\n\n"
        prompt += f"Speaker B:\n{str(row['transcript_b'])[:2000]}\n"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip().lower()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    preds = []
    for line in lines:
        preds.append(line)

    while len(preds) < len(rows):
        preds.append("MISSING_OUTPUT")

    return preds[:len(rows)]

def load_log():
    prediction_map = {}
    if not os.path.exists(LOG_FILE):
        return prediction_map
    with open(LOG_FILE, "r") as f:
        for line in f:
            if "|" not in line:
                continue
            parts = line.strip().split("|")
            filename = parts[0].strip()
            pred = parts[-1].replace("PRED:", "").strip()
            prediction_map[filename] = pred
    return prediction_map

def atomic_rewrite_log(prediction_map, df):
    temp_file = LOG_FILE + ".tmp"
    with open(temp_file, "w") as f:
        for filename, pred in prediction_map.items():
            matches = df[df["mixed_audio_filename"] == filename]
            if not matches.empty:
                true_label = matches["relationship_detail_a"].values[0]
                f.write(f"{filename} | TRUE: {true_label} | PRED: {pred}\n")
    os.replace(temp_file, LOG_FILE)

def build_output_csv(prediction_map, df):
    df["llm_predicted_relationship"] = df["mixed_audio_filename"].map(prediction_map)

    def evaluate(row):
        pred = row["llm_predicted_relationship"]
        true = row["relationship_detail_a"]
        if pred in VALID_LABELS and pred == true:
            return "llm-guessed-correct"
        if pred in VALID_LABELS and pred != true:
            return "llm-guessed-wrong"
        return "llm-off-label"

    df["llm_text_evaluation"] = df.apply(evaluate, axis=1)
    df.to_csv(OUTPUT_CSV, index=False)

    valid_rows = df[
        df["relationship_detail_a"].isin(VALID_LABELS) &
        df["llm_predicted_relationship"].isin(VALID_LABELS)
    ]

    y_true = valid_rows["relationship_detail_a"]
    y_pred = valid_rows["llm_predicted_relationship"]

    print("\nMacro F1:", f1_score(y_true, y_pred, average="macro"))
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    df_raw = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df_raw[df_raw['label'] == 'naturalistic'].copy()
    
    df["relationship_detail_a"] = df["relationship_detail_a"].astype(str).str.strip()

    prediction_map = load_log()

    missing_rows = [
        row for _, row in df.iterrows()
        if row["mixed_audio_filename"] not in prediction_map
    ]

    print(f"Phase 1 — Missing from log: {len(missing_rows)}")

    if missing_rows:
        with open(LOG_FILE, "a") as log_file:
            for i in range(0, len(missing_rows), BATCH_SIZE):
                batch = missing_rows[i:i+BATCH_SIZE]
                preds = classify_batch(batch)

                for row, pred in zip(batch, preds):
                    filename = row["mixed_audio_filename"]
                    true_label = row["relationship_detail_a"]
                    prediction_map[filename] = pred
                    log_file.write(f"{filename} | TRUE: {true_label} | PRED: {pred}\n")

                print(f"Coverage progress: {min(i+BATCH_SIZE, len(missing_rows))}/{len(missing_rows)}")

    invalid_rows = [
        row for _, row in df.iterrows()
        if prediction_map.get(row["mixed_audio_filename"]) not in VALID_LABELS
        or prediction_map.get(row["mixed_audio_filename"]) in ["MISSING_OUTPUT"]
    ]

    print(f"\nPhase 2 — Invalid rows to fix: {len(invalid_rows)}")


    if invalid_rows:
        for i in range(0, len(invalid_rows), BATCH_SIZE):
            batch = invalid_rows[i:i+BATCH_SIZE]
            preds = classify_batch(batch)

            for row, pred in zip(batch, preds):
                prediction_map[row["mixed_audio_filename"]] = pred

            if (i // BATCH_SIZE) % 10 == 0 or i + BATCH_SIZE >= len(invalid_rows):
                atomic_rewrite_log(prediction_map, df)
                print(f"Repair progress: {min(i+BATCH_SIZE, len(invalid_rows))}/{len(invalid_rows)} (saved)")
            else:
                print(f"Repair progress: {min(i+BATCH_SIZE, len(invalid_rows))}/{len(invalid_rows)}")

    remaining_invalid = [
        k for k, v in prediction_map.items() 
        if k in df["mixed_audio_filename"].values and v not in VALID_LABELS
    ]

    if remaining_invalid:
        print(f"\nStill invalid rows remaining: {len(remaining_invalid)}")
        return

    print("\nAll rows valid. Building final CSV.")
    build_output_csv(prediction_map, df)
    print("Done.")

if __name__ == "__main__":
    main()