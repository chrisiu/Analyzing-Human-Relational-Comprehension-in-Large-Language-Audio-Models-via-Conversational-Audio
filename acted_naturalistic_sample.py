import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "audiofiles_transcripts_merged.csv"
df = pd.read_csv(INPUT_PATH, low_memory=False)

nat_df = df[df['label'] == 'naturalistic'].copy()
imp_df = df[df['label'] == 'improvised'].copy()

counts = imp_df['acted_relationship_a'].value_counts()
single_member_classes = counts[counts < 2].index.tolist()
if single_member_classes:
    print(f"Removing rare improvised classes with < 2 members: {single_member_classes}")
    imp_df = imp_df[~imp_df['acted_relationship_a'].isin(single_member_classes)]

protected_labels = ["neighbors", "roommates"]
rare_nat = nat_df[nat_df['relationship_detail_a'].isin(protected_labels)]
main_nat = nat_df[~nat_df['relationship_detail_a'].isin(protected_labels)]

needed_nat = 750 - len(rare_nat)

_, subset_nat = train_test_split(
    main_nat,
    test_size=needed_nat,
    stratify=main_nat['relationship_detail_a'],
    random_state=42
)
final_nat = pd.concat([rare_nat, subset_nat])


_, final_imp = train_test_split(
    imp_df,
    test_size=750,
    stratify=imp_df['acted_relationship_a'],
    random_state=42
)

final_subset = pd.concat([final_nat, final_imp]).sample(frac=1, random_state=42)

print(f"Total Rows: {len(final_subset)}")
print(f"Naturalistic Count: {len(final_nat)} | Improvised Count: {len(final_imp)}")

OUTPUT_PATH = "lalm_nat_imp_relationships.csv"
final_subset.to_csv(OUTPUT_PATH, index=False)
print(f"\nFile saved successfully to: {OUTPUT_PATH}")
