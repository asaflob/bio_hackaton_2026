import pandas as pd
import numpy as np

# Load datasets
df_neg = pd.read_csv('negatives_with_secondary_sliced.csv')
df_pas = pd.read_csv('nes_pattern_location.csv')

# --- Process Negative Dataset ---

# Rename 'secondary_seq' to 'secondary' to match the desired format
if 'secondary_seq' in df_neg.columns:
    df_neg = df_neg.rename(columns={'secondary_seq': 'secondary'})

# Select only the relevant columns
# We use the existing 'secondary' column from the file, no new column is created
df_neg_clean = df_neg[['uniprotID', 'sequence', 'secondary', 'label']]

# --- Process Positive Dataset ---

# Drop the original 'sequence' column if it exists to avoid conflicts
if 'sequence' in df_pas.columns:
    df_pas = df_pas.drop(columns=['sequence'])

# Rename 'aa_sequence' to 'sequence' and assign label 1 for positives
df_pas = df_pas.rename(columns={'aa_sequence': 'sequence'})
df_pas['label'] = 1

# Select only the relevant columns
df_pas_clean = df_pas[['uniprotID', 'sequence', 'secondary', 'label']]

# --- Combine and Shuffle ---

# Concatenate both dataframes
df_combined = pd.concat([df_neg_clean, df_pas_clean], ignore_index=True)

# Shuffle the combined dataset
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Save Output ---
output_filename = 'combine_data.csv'
df_combined.to_csv(output_filename, index=False)

print(f"Successfully saved {len(df_combined)} samples to '{output_filename}'")
print(df_combined.head())