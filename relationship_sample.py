import pandas as pd

import pandas as pd
import os

INPUT_PATH = "audiofiles_gpt5_eval.csv"
OUTPUT_PATH = "lalm_relationships.csv"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input file not found at {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, low_memory=False)

correct_pool = df[df['llm_text_evaluation'] == 'llm-guessed-correct'].copy()
wrong_pool = df[df['llm_text_evaluation'] == 'llm-guessed-wrong'].copy()

def balanced_sample(data, target_total):
    counts = data['relationship_detail_a'].value_counts()
    groups = data.groupby('relationship_detail_a')
    
    per_class_target = target_total // len(counts)
    
    samples = []
    for name, group in groups:
        n = min(len(group), per_class_target)
        samples.append(group.sample(n=n, random_state=42))
    
    combined = pd.concat(samples)
    
    remaining = data.drop(combined.index)
    needed = target_total - len(combined)
    if needed > 0:
        extra = remaining.sample(n=needed, random_state=42)
        combined = pd.concat([combined, extra])
        
    return combined

final_correct = balanced_sample(correct_pool, 750)
final_wrong = balanced_sample(wrong_pool, 750)

final_subset = pd.concat([final_correct, final_wrong]).sample(frac=1, random_state=42)

split_df = pd.DataFrame({
    'Correct': final_correct['relationship_detail_a'].value_counts(),
    'Wrong': final_wrong['relationship_detail_a'].value_counts()
}).fillna(0).astype(int)

print(split_df)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_subset.to_csv(OUTPUT_PATH, index=False)