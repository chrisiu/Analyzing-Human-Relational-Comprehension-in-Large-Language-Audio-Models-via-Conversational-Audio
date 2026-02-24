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

merged_with_sessions = files.merge(
    relationships, 
    on=['vendor_id', 'session_id'], 
    how='left'
)

fully_merged = merged_with_sessions.merge(
    prompts, 
    left_on='interaction_hash', 
    right_on='prompt_hash', 
    how='left'
)

output_path = 'fully_merged_dataset.csv'
fully_merged.to_csv(output_path, index=False)
