import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_model_data():
    # 1. טעינת הדאטה המאוחד
    print("Loading combined dataset...")
    df = pd.read_csv('combine_data.csv')

    # 2. פיצול ראשוני ל-Train ו-Test (לפני האיזון!)
    # אנו שומרים על Stratify כדי שה-Test יישאר עם היחס המקורי (הקשה) של העולם האמיתי
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        df.drop('label', axis=1),
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # איחוד חזרה ל-DataFrame כדי שיהיה נוח לסנן
    train_set_raw = pd.concat([X_train_raw, y_train_raw], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    # 3. איזון ה-Train Set ליחס 1:10
    print("\nBalancing Train set...")

    # הפרדה לחיוביים ושליליים בתוך ה-Train
    train_pos = train_set_raw[train_set_raw['label'] == 1]
    train_neg = train_set_raw[train_set_raw['label'] == 0]

    # חישוב יעד השליליים (פי 10 מהחיוביים)
    num_pos = len(train_pos)
    target_neg_count = num_pos * 10

    # בדיקת תקינות (למקרה שאין מספיק שליליים, מה שלא סביר שיקרה אצלך)
    if target_neg_count > len(train_neg):
        target_neg_count = len(train_neg)
        print("Warning: Not enough negatives for 1:10 ratio, taking all available.")

    # דגימה אקראית של השליליים
    train_neg_balanced = train_neg.sample(n=target_neg_count, random_state=42)

    # איחוד מחדש של ה-Train המאוזן
    train_set_final = pd.concat([train_pos, train_neg_balanced])

    # ערבוב סופי (Shuffle) של ה-Train וה-Test
    train_set_final = train_set_final.sample(frac=1, random_state=42).reset_index(drop=True)
    test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. שמירת הקבצים
    train_set_final.to_csv('train_set_balanced.csv', index=False)
    test_set.to_csv('test_set.csv', index=False)

    # הדפסת סטטיסטיקות
    print("\n--- Summary ---")
    print(f"Original Train size: {len(train_set_raw)}")
    print(f"Balanced Train size: {len(train_set_final)}")
    print(f"  - Positives: {len(train_pos)}")
    print(f"  - Negatives: {len(train_neg_balanced)} (Ratio 1:{len(train_neg_balanced) / len(train_pos):.1f})")
    print(f"Test Set size: {len(test_set)} (Original Distribution)")
    print("Files saved: 'train_set_balanced.csv', 'test_set.csv'")


if __name__ == "__main__":
    prepare_model_data()