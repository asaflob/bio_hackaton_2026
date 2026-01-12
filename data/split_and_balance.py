MAX_NEGATIVE = 2
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_model_data():
    # 1. Load the combined dataset
    print("Loading combined dataset...")
    df = pd.read_csv('combine_data.csv')

    # 2. Separate Positives and Negatives
    positives = df[df['label'] == 1]
    negatives = df[df['label'] == 0]

    print(f"Total available positives: {len(positives)}")
    print(f"Total available negatives: {len(negatives)}")

    # 3. Balance the entire dataset FIRST
    # We want to keep ALL positives and sample negatives to match the ratio
    num_pos = len(positives)
    target_neg_count = num_pos * MAX_NEGATIVE

    # Check if we have enough negatives
    if target_neg_count > len(negatives):
        print(f"Warning: Not enough negatives for 1:{MAX_NEGATIVE} ratio. Using all available negatives.")
        target_neg_count = len(negatives)

    print(f"\nBalancing dataset to 1:{MAX_NEGATIVE} ratio...")

    # Sample the negatives
    negatives_balanced = negatives.sample(n=target_neg_count, random_state=42)

    # Combine positives and sampled negatives
    df_balanced = pd.concat([positives, negatives_balanced])

    # 4. Split into Train and Test
    # We use 'stratify' to ensure both Train and Test get the exact same 1:MAX_NEGATIVE ratio
    train_set, test_set = train_test_split(
        df_balanced,
        test_size=0.2,
        random_state=42,
        stratify=df_balanced['label']
    )

    # 5. Final Shuffle (Just to be safe, though train_test_split usually shuffles)
    train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
    test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Save files
    train_set.to_csv('train_set_balanced.csv', index=False)
    test_set.to_csv('test_set.csv', index=False)

    # --- Summary ---
    print("\n--- Summary ---")

    # Helper function to calculate ratio
    def get_stats(data, name):
        n_pos = len(data[data['label'] == 1])
        n_neg = len(data[data['label'] == 0])
        ratio = n_neg / n_pos if n_pos > 0 else 0
        print(f"{name} size: {len(data)}")
        print(f"  - Positives: {n_pos}")
        print(f"  - Negatives: {n_neg} (Ratio 1:{ratio:.1f})")

    get_stats(train_set, "Train Set")
    get_stats(test_set, "Test Set")

    print("\nFiles saved: 'train_set_balanced.csv', 'test_set.csv'")


if __name__ == "__main__":
    prepare_model_data()