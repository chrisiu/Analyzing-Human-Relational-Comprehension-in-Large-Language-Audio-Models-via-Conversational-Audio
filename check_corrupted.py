import pandas as pd
from pathlib import Path
import shutil

corrupted_files = [
    "V00_S0693_I00000375_mixed.wav", "V03_S0081_I00000134_mixed.wav", "V03_S0693_I00000400_mixed.wav",
    "V00_S0693_I00000373_mixed.wav", "V00_S0761_I00000373_mixed.wav", "V00_S0761_I00000377_mixed.wav",
    "V03_S0084_I00000140_mixed.wav", "V03_S0877_I00000423_mixed.wav", "V01_S0099_I00000140_mixed.wav",
    "V02_S3640_I00000212_mixed.wav", "V03_S0051_I00000382_mixed.wav", "V01_S0104_I00000125_mixed.wav",
    "V01_S0099_I00000387_mixed.wav", "V02_S3640_I00000217_mixed.wav", "V03_S0646_I00000118_mixed.wav",
    "V00_S0693_I00000374_mixed.wav", "V03_S0051_I00000384_mixed.wav", "V03_S0081_I00000128_mixed.wav",
    "V01_S0104_I00000126_mixed.wav", "V03_S0070_I00000309_mixed.wav", "V03_S0051_I00000375_mixed.wav",
    "V00_S0761_I00000381_mixed.wav", "V02_S3367_I00000397_mixed.wav", "V01_S0338_I00001111_mixed.wav",
    "V01_S0099_I00000313_mixed.wav", "V01_S0104_I00000130_mixed.wav", "V03_S1563_I00000053_mixed.wav",
    "V01_S0104_I00000499_mixed.wav", "V03_S0084_I00000134_mixed.wav", "V00_S0512_I00000500_mixed.wav",
    "V03_S0084_I00000139_mixed.wav", "V00_S0224_I00000482_mixed.wav", "V03_S0051_I00000385_mixed.wav",
    "V01_S0635_I00001240_mixed.wav", "V03_S2003_I00000220_mixed.wav", "V01_S0104_I00000461_mixed.wav",
    "V03_S2028_I00000246_mixed.wav", "V02_S3367_I00000396_mixed.wav", "V03_S0646_I00000119_mixed.wav",
    "V02_S3367_I00000438_mixed.wav", "V03_S0196_I00000489_mixed.wav", "V03_S2003_I00000219_mixed.wav",
    "V03_S1563_I00000024_mixed.wav", "V03_S0081_I00000130_mixed.wav", "V03_S0051_I00000138_mixed.wav",
    "V03_S2003_I00000211_mixed.wav", "V03_S0070_I00000134_mixed.wav", "V03_S0154_I00000387_mixed.wav",
    "V03_S2003_I00000222_mixed.wav", "V03_S0193_I00000498_mixed.wav", "V00_S0693_I00000377_mixed.wav",
    "V03_S0070_I00000137_mixed.wav", "V02_S3640_I00000330_mixed.wav", "V02_S3640_I00000215_mixed.wav",
    "V03_S0154_I00000498_mixed.wav", "V03_S1038_I00000490_mixed.wav", "V03_S0161_I00000480_mixed.wav",
    "V03_S2003_I00000217_mixed.wav", "V02_S5031_I00000305_mixed.wav", "V03_S0051_I00000502_mixed.wav",
    "V03_S0242_I00000383_mixed.wav", "V03_S0051_I00000377_mixed.wav", "V02_S3640_I00000331_mixed.wav",
    "V00_S0693_I00000379_mixed.wav", "V03_S2071_I00000101_mixed.wav", "V00_S0224_I00000477_mixed.wav",
    "V03_S0154_I00000313_mixed.wav", "V03_S1038_I00000377_mixed.wav", "V03_S0777_I00000120_mixed.wav",
    "V03_S1563_I00000051_mixed.wav", "V00_S0512_I00000489_mixed.wav", "V00_S0693_I00000372_mixed.wav",
    "V03_S0233_I00000498_mixed.wav", "V03_S0083_I00000499_mixed.wav", "V00_S0224_I00000480_mixed.wav",
    "V03_S0193_I00000386_mixed.wav", "V02_S3520_I00000332_mixed.wav", "V00_S0224_I00000483_mixed.wav",
    "V03_S2003_I00000215_mixed.wav", "V02_S5031_I00000306_mixed.wav", "V03_S1167_I00000053_mixed.wav",
    "V03_S2071_I00000100_mixed.wav", "V01_S0346_I00000725_mixed.wav", "V00_S0512_I00000495_mixed.wav",
    "V00_S0761_I00000371_mixed.wav", "V02_S3640_I00000216_mixed.wav", "V03_S0081_I00000139_mixed.wav",
    "V03_S0081_I00000309_mixed.wav", "V00_S0512_I00000482_mixed.wav", "V03_S0507_I00000435_mixed.wav",
    "V03_S0242_I00000385_mixed.wav", "V03_S0242_I00000386_mixed.wav", "V01_S0104_I00000501_mixed.wav",
    "V03_S1055_I00000304_mixed.wav", "V03_S2003_I00000332_mixed.wav", "V02_S3640_I00000220_mixed.wav",
    "V01_S0099_I00000498_mixed.wav", "V02_S3640_I00000223_mixed.wav", "V03_S0051_I00000504_mixed.wav",
    "V03_S2071_I00000099_mixed.wav", "V03_S0938_I00000178_mixed.wav", "V03_S0914_I00000440_mixed.wav",
    "V03_S2003_I00000221_mixed.wav", "V02_S3367_I00000394_mixed.wav", "V01_S0104_I00000462_mixed.wav",
    "V03_S0083_I00000313_mixed.wav", "V02_S3367_I00000398_mixed.wav", "V03_S0084_I00000137_mixed.wav",
    "V00_S0512_I00000514_mixed.wav", "V03_S0651_I00000207_mixed.wav", "V03_S0242_I00000382_mixed.wav",
    "V01_S0104_I00000128_mixed.wav", "V03_S0081_I00000126_mixed.wav", "V00_S0761_I00000379_mixed.wav",
    "V03_S0081_I00000138_mixed.wav", "V03_S1038_I00000491_mixed.wav", "V03_S2071_I00000230_mixed.wav",
    "V03_S0242_I00000384_mixed.wav", "V02_S3640_I00000332_mixed.wav", "V03_S2003_I00000218_mixed.wav",
    "V01_S0104_I00000138_mixed.wav", "V03_S0161_I00000500_mixed.wav", "V03_S0161_I00000484_mixed.wav",
    "V03_S0193_I00000384_mixed.wav", "V01_S0104_I00000498_mixed.wav", "V02_S3640_I00000213_mixed.wav",
    "V01_S0104_I00000132_mixed.wav", "V03_S2003_I00000223_mixed.wav", "V00_S0224_I00000479_mixed.wav",
    "V03_S2003_I00000212_mixed.wav", "V03_S0232_I00000132_mixed.wav", "V03_S0084_I00000309_mixed.wav",
    "V02_S3640_I00000211_mixed.wav", "V03_S0161_I00000487_mixed.wav", "V02_S3367_I00000400_mixed.wav",
    "V01_S0635_I00001222_mixed.wav", "V03_S2003_I00000331_mixed.wav", "V03_S0161_I00000377_mixed.wav",
    "V02_S3640_I00000221_mixed.wav", "V03_S0242_I00000504_mixed.wav", "V00_S0761_I00000375_mixed.wav",
    "V00_S0761_I00000378_mixed.wav", "V02_S3640_I00000219_mixed.wav", "V03_S0081_I00000131_mixed.wav",
    "V02_S3367_I00000395_mixed.wav", "V01_S0473_I00001140_mixed.wav", "V00_S0512_I00000478_mixed.wav",
    "V03_S2003_I00000216_mixed.wav", "V02_S3640_I00000222_mixed.wav", "V03_S0051_I00000383_mixed.wav",
    "V00_S0224_I00000478_mixed.wav", "V01_S1085_I00001238_mixed.wav", "V03_S0070_I00000132_mixed.wav",
    "V03_S0161_I00000510_mixed.wav", "V03_S0193_I00000385_mixed.wav", "V02_S3367_I00000440_mixed.wav",
    "V00_S0512_I00000483_mixed.wav", "V00_S0224_I00000495_mixed.wav", "V03_S0161_I00000483_mixed.wav",
    "V03_S0081_I00000137_mixed.wav", "V03_S1038_I00000510_mixed.wav", "V00_S0512_I00000484_mixed.wav",
    "V03_S0161_I00000491_mixed.wav", "V02_S3520_I00000330_mixed.wav", "V03_S1563_I00000052_mixed.wav",
    "V03_S1038_I00000489_mixed.wav", "V02_S3640_I00000218_mixed.wav", "V00_S0512_I00000515_mixed.wav",
    "V02_S3367_I00000399_mixed.wav", "V01_S0635_I00001236_mixed.wav", "V03_S0084_I00000138_mixed.wav",
    "V00_S0693_I00000378_mixed.wav", "V03_S2028_I00000249_mixed.wav", "V03_S1213_I00000132_mixed.wav",
    "V03_S0777_I00000122_mixed.wav", "V03_S0693_I00000440_mixed.wav", "V00_S0512_I00000486_mixed.wav",
    "V03_S0084_I00000313_mixed.wav", "V02_S3367_I00000439_mixed.wav", "V03_S0154_I00000140_mixed.wav",
    "V00_S0761_I00000372_mixed.wav", "V03_S0693_I00000439_mixed.wav", "V03_S0161_I00000488_mixed.wav",
    "V03_S0161_I00000490_mixed.wav", "V00_S0224_I00000484_mixed.wav", "V03_S0193_I00000387_mixed.wav",
    "V03_S0161_I00000482_mixed.wav", "V03_S0193_I00000140_mixed.wav", "V03_S1055_I00000303_mixed.wav",
    "V01_S0099_I00000139_mixed.wav", "V03_S2003_I00000213_mixed.wav", "V01_S0104_I00000309_mixed.wav",
    "V03_S0242_I00000140_mixed.wav", "V03_S2028_I00000247_mixed.wav"
]

files_to_clean = [
    "audiofiles_transcripts.csv",
]

print("\nREMOVING CORRUPTED AUDIO FILES\n")
print(f"\nTotal corrupted files to remove: {len(corrupted_files)}\n")

for file_path in files_to_clean:
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"    File not found: {file_path}")
        print(f"    Skipping.\n")
        continue
    
    print(f"Processing: {file_path.name}")
    
    backup_path = file_path.parent / f"{file_path.stem}_backup{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    print(f"  ✓ Backup created: {backup_path.name}")
    
    try:
        df = pd.read_csv(file_path)
        original_count = len(df)
        print(f"  Original rows: {original_count}")
        
        audio_col = None
        for col in df.columns:
            if 'audio' in col.lower() and 'filename' in col.lower():
                audio_col = col
                break
        
        if audio_col is None:
            possible_cols = ['mixed_audio_filename', 'audio_filename', 'filename', 'file']
            for col in possible_cols:
                if col in df.columns:
                    audio_col = col
                    break
        
        if audio_col is None:
            print(f"     Could not find audio filename column")
            print(f"     Skipping.\n")
            continue
        
        print(f"  Using column: {audio_col}")
        
        df_clean = df[~df[audio_col].isin(corrupted_files)]
        removed_count = original_count - len(df_clean)
        
        df_clean.to_csv(file_path, index=False)
        
        print(f"  ✓ Removed {removed_count} corrupted rows")
        print(f"  ✓ Cleaned rows: {len(df_clean)}")
        print(f"  ✓ Saved: {file_path.name}\n")


