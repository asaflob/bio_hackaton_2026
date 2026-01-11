import pandas as pd
import io
import numpy as np

# קריאת הנתונים
df_neg = pd.read_csv('negatives_dataset.csv')
df_pas = pd.read_csv('nes_pattern_location.csv')

# --- עיבוד הטבלה השלילית (Negative) ---
#df_neg = df_neg.rename(columns={'neuniprotID': 'uniprotID'})#

# בדיקה האם קיימת עמודת secondary
if 'secondary' not in df_neg.columns:
    df_neg['secondary'] = df_neg['sequence'].apply(lambda x: 'C' * len(x))

# בחירת העמודות הרלוונטיות
df_neg_clean = df_neg[['uniprotID', 'sequence', 'secondary', 'label']]


# --- עיבוד הטבלה החיובית (Positive) ---
# הסרת עמודת sequence המקורית (המלוכלכת) אם קיימת, כדי למנוע התנגשות
if 'sequence' in df_pas.columns:
    df_pas = df_pas.drop(columns=['sequence'])

# שינוי השם והגדרת label
df_pas = df_pas.rename(columns={'aa_sequence': 'sequence'})
df_pas['label'] = 1

# בחירת העמודות הרלוונטיות
df_pas_clean = df_pas[['uniprotID', 'sequence', 'secondary', 'label']]


# --- איחוד וערבוב ---
df_combined = pd.concat([df_neg_clean, df_pas_clean], ignore_index=True)

# ערבוב (Shuffle)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# --- שמירת הקובץ ---
output_filename = 'combine_data.csv'
df_combined.to_csv(output_filename, index=False)

print(f"Successfully saved {len(df_combined)} samples to '{output_filename}'")
print(df_combined.head())