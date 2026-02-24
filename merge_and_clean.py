import pandas as pd
import os

files = pd.read_csv('filelist.csv')
prompts = pd.read_csv('interactions.csv') 
relationships = pd.read_csv('relationships.csv')

def parse_file_id(file_id):
    parts = file_id.split('_')
    return {
        'vendor_id': parts[0],
        'session_id': int(parts[1].replace('S', '')),
        'interaction_id': parts[2],
        'participant_id': parts[3]
    }

parsed = files['file_id'].apply(parse_file_id).apply(pd.Series)
files = pd.concat([files, parsed], axis=1)
files['interaction_hash'] = files['interaction_id'].str.replace('I', '').str.zfill(8).astype(str)
prompts['prompt_hash'] = prompts['prompt_hash'].astype(str).str.zfill(8)

merged_base = files.merge(relationships, on=['vendor_id', 'session_id'], how='left')
fully_merged = merged_base.merge(prompts, left_on='interaction_hash', right_on='prompt_hash', how='left')

fully_merged = fully_merged.drop(columns=[c for c in fully_merged.columns if c.startswith("Unnamed")], errors="ignore")
fully_merged = fully_merged.replace({r"\r\n|\r|\n": " "}, regex=True)
fully_merged.to_csv('fully_merged_dataset.csv', index=False)

def has_improvised_format(p_a, p_b):
    check = lambda p: isinstance(p, str) and p.strip().startswith("(") and "): As" in p
    return check(p_a) or check(p_b)

df = fully_merged[fully_merged["interaction_type"] == "ipc_conversation"].copy()
df = df.replace(r"^\s*$", pd.NA, regex=True)
df["interaction_key"] = df["file_id"].str.split("_").str[:3].str.join("_")

interaction_rows = []
for key, g in df.groupby("interaction_key"):
    if len(g) == 2:
        g = g.sort_values("file_id").reset_index(drop=True)
        a, b = g.iloc[0], g.iloc[1]
        
        if a["label"] == "naturalistic" and (pd.isna(a["relationship_detail"]) or pd.isna(b["relationship_detail"])):
            continue
        if a["label"] == "improvised" and not has_improvised_format(a["participant_a_prompt_text"], b["participant_b_prompt_text"]):
            continue

        interaction_rows.append({
            "mixed_audio_filename": f"{key}_mixed.wav",
            "file_id_a": a["file_id"], "file_id_b": b["file_id"],
            "label": a["label"],
            "relationship_detail_a": a["relationship_detail"], "relationship_detail_b": b["relationship_detail"],
            "ipc_a": a["ipc_a"], "ipc_b": b["ipc_b"],
            "participant_a_prompt_text": a["participant_a_prompt_text"],
            "participant_b_prompt_text": b["participant_b_prompt_text"]
        })

pd.DataFrame(interaction_rows).to_csv('fully_merged_dataset.csv', index=False)
