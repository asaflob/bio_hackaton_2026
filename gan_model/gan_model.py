import pandas as pd
import torch
import requests
import matplotlib.pyplot as plt
import time
import numpy as np
from transformers import AutoTokenizer, EsmModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def get_protein_sequence(uniprot_id):
    """
    שולף את רצף החלבון המלא מ-UniProt לפי ID
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.ok:
        lines = response.text.split('\n')
        sequence = ''.join(lines[1:])
        return sequence
    else:
        return None

def get_window_embedding(full_seq, start_idx, end_idx):
    """
    מקבל חלבון מלא, מריץ ESM-2, וגוזר את הווקטור הממוצע של החלון הספציפי
    """
    # 1. הכנת הקלט למודל
    inputs = tokenizer(full_seq, return_tensors="pt", add_special_tokens=False)

    # אם החלבון ארוך מדי למודל, אפשר לחתוך, אבל לצורך הבדיקה נניח שזה נכנס
    # (המודל הזה תומך עד 1024 בדרך כלל, תלוי בקונפיגורציה)

    with torch.no_grad():
        outputs = model(**inputs)

    # 2. שליפת ה-Embeddings (המצב הנסתר האחרון)
    # Shape: (1, Sequence_Length, 320)
    embeddings = outputs.last_hidden_state[0]

    # 3. חיתוך החלון הרלוונטי
    # שים לב: המודל עשוי להוסיף טוקנים מיוחדים בהתחלה/סוף אם add_special_tokens=True.
    # כאן שמנו False, אבל צריך לוודא אינדקסים.

    # הגנה מפני חריגה
    s = max(0, start_idx)
    e = min(len(embeddings), end_idx)

    window_emb = embeddings[s:e]

    # 4. ממוצע (Mean Pooling) כדי לקבל וקטור אחד בגודל 320
    mean_emb = torch.mean(window_emb, dim=0)
    return mean_emb.numpy()

def ESM_model_running():
    # --- ביצוע הבדיקה ---

    # טען את הקבצים
    df_pos = pd.read_csv('nes_pattern_location.csv')  # החיוביים
    df_neg = pd.read_csv('negatives_dataset.csv')  # השליליים

    # בחר 3 חלבונים שיש להם גם חיובי וגם שלילי
    test_ids = list(set(df_pos['uniprotID']).intersection(set(df_neg['uniprotID'])))[:50]

    print(f"\nAnalyzing proteins: {test_ids}")

    for pid in test_ids:
        print(f"\n--- Protein: {pid} ---")

        # 1. שליפת הרצף המלא
        full_seq = get_protein_sequence(pid)
        if not full_seq:
            print("Skipping (No sequence)")
            continue

        vectors = []
        labels = []  # 'Positive' or 'Negative'
        colors = []

        # 2. עיבוד ה-NES האמיתי (Positive)
        pos_row = df_pos[df_pos['uniprotID'] == pid].iloc[0]  # ניקח את הראשון
        try:
            # המרה לאינדקס 0-based
            p_start = int(pos_row['start#']) - 1
            p_end = p_start + len(pos_row['aa_sequence'])

            vec_pos = get_window_embedding(full_seq, p_start, p_end)
            vectors.append(vec_pos)
            labels.append("True NES")
            colors.append('blue')
            print(f"Encoded True NES (Length {p_end - p_start})")
        except Exception as e:
            print(f"Error encoding positive: {e}")
            continue

        # 3. עיבוד השליליים (Negatives)
        neg_rows = df_neg[df_neg['uniprotID'] == pid].head(10)  # ניקח 10 דוגמאות
        for _, row in neg_rows.iterrows():
            n_start = int(row['start_index']) - 1
            n_end = int(row['end_index'])

            try:
                vec_neg = get_window_embedding(full_seq, n_start, n_end)
                vectors.append(vec_neg)
                labels.append("Negative")
                colors.append('red')
            except:
                pass
        print(f"Encoded {len(neg_rows)} Negatives")

        # 4. ויזואליזציה עם PCA (הורדה ל-2 מימדים)
        if len(vectors) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(np.array(vectors))

            plt.figure(figsize=(8, 6))

            # ציור השליליים (אדום)
            neg_indices = [i for i, label in enumerate(labels) if label == "Negative"]
            plt.scatter(X_pca[neg_indices, 0], X_pca[neg_indices, 1], c='red', alpha=0.6, label='Negative')

            # ציור החיובי (כוכב כחול גדול)
            pos_indices = [i for i, label in enumerate(labels) if label == "True NES"]
            plt.scatter(X_pca[pos_indices, 0], X_pca[pos_indices, 1], c='blue', s=200, marker='*', label='True NES')

            plt.title(f"ESM-2 Embedding Space for {pid}\n(PCA Reduction)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # חישוב מרחקים (Cosine)
            # נבדוק מה המרחק בין ה-NES לבין ממוצע השליליים
            true_vec = vectors[0].reshape(1, -1)
            neg_vecs = np.array(vectors[1:])
            similarities = cosine_similarity(true_vec, neg_vecs)

            print(f"Avg Cosine Similarity to Negatives: {np.mean(similarities):.4f}")
            print(f"Max Similarity (closest negative): {np.max(similarities):.4f}")
            print(f"Min Similarity (furthest negative): {np.min(similarities):.4f}")

        else:
            print("Not enough vectors to plot.")


if __name__ == '__main__':
    MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    print(f"Loading ESM-2 Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()  # מצב Evaluation (לא אימון)




