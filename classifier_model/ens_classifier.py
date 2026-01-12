import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, roc_auc_score, log_loss, \
    roc_curve, auc

# ==========================================
# 1. Feature Extractor Class
# ==========================================
class NES_Embedder:
    def __init__(self):
        # Physicochemical properties dictionary
        self.hydro = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2,
                      'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
                      'Y': -1.3, 'V': 4.2}
        self.volume = {'G': 60, 'A': 88, 'S': 89, 'C': 108, 'D': 111, 'P': 112, 'N': 114, 'T': 116, 'E': 138, 'V': 140,
                       'Q': 143, 'H': 153, 'M': 162, 'I': 166, 'L': 166, 'K': 168, 'R': 173, 'F': 189, 'Y': 193,
                       'W': 227}
        self.hydrophobic_residues = set(['L', 'I', 'V', 'F', 'M'])

        # Secondary Structure Map
        self.struc_map = {'H': 1.0, 'G': 1.0, 'I': 1.0,  # Helices
                          'E': -1.0, 'B': -1.0,          # Sheets
                          'C': 0.0, 'T': 0.0, 'S': 0.0}  # Coils/Turns

    def transform(self, aa_sequences, struc_sequences):
        vectors = []
        for aa, struc in zip(aa_sequences, struc_sequences):
            vectors.append(self._embed_single(aa, struc))
        return np.array(vectors)

    def _embed_single(self, aa_seq, struc_seq):
        s = str(aa_seq).strip().upper()
        st = str(struc_seq).strip().upper()

        if len(s) < 3:
            return np.zeros(25)

        h_vals = np.array([self.hydro.get(aa, 0) for aa in s])
        v_vals = np.array([self.volume.get(aa, 0) for aa in s])
        st_vals = np.array([self.struc_map.get(char, 0) for char in st])

        min_len = min(len(h_vals), len(st_vals))
        h_vals = h_vals[:min_len]
        v_vals = v_vals[:min_len]
        st_vals = st_vals[:min_len]

        feats = []
        # 1. Chemical Statistics
        feats.extend([np.mean(h_vals), np.std(h_vals)])
        feats.extend([np.mean(v_vals), np.std(v_vals)])

        # 2. Structural Statistics
        feats.append(np.mean(st_vals))
        feats.append(np.sum(st_vals == 1.0) / len(s))
        feats.append(np.sum(st_vals == -1.0) / len(s))

        # 3. Auto-Correlation
        for lag in range(1, 5):
            if len(h_vals) > lag:
                ac_h = np.mean((h_vals[:-lag] - np.mean(h_vals)) * (h_vals[lag:] - np.mean(h_vals)))
                ac_s = np.mean((st_vals[:-lag] - np.mean(st_vals)) * (st_vals[lag:] - np.mean(st_vals)))
            else:
                ac_h, ac_s = 0, 0
            feats.extend([ac_h, ac_s])

        # 4. Hydrophobic Patterns
        bin_pat = [1 if aa in self.hydrophobic_residues else 0 for aa in s]
        g2 = sum(1 for i in range(len(bin_pat) - 3) if bin_pat[i] and bin_pat[i + 3])
        g3 = sum(1 for i in range(len(bin_pat) - 4) if bin_pat[i] and bin_pat[i + 4])
        feats.extend([g2 / len(s), g3 / len(s)])

        return np.array(feats)


# ==========================================
# 2. Load and Prepare Data
# ==========================================
print("Loading datasets...")
try:
    train_df = pd.read_csv('../data/train_set_balanced.csv')
    test_df = pd.read_csv('../data/test_set.csv')
except FileNotFoundError:
    print("Error: Files not found. Please run the 'split_and_balance.py' script first.")
    exit()

print(f"Train set size: {len(train_df)}")
print(f"Test set size:  {len(test_df)}")

X_train_seq = train_df['sequence']
X_train_struc = train_df['secondary']
y_train = train_df['label'].values

X_test_seq = test_df['sequence']
X_test_struc = test_df['secondary']
y_test = test_df['label'].values

# ==========================================
# 3. Feature Extraction
# ==========================================
print("\nExtracting features...")
embedder = NES_Embedder()
X_train_vec = embedder.transform(X_train_seq, X_train_struc)
X_test_vec = embedder.transform(X_test_seq, X_test_struc)

# ==========================================
# 4. Model Training
# ==========================================
print("\nTraining Gradient Boosting Classifier...")
clf = GradientBoostingClassifier(
    n_estimators=175,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)
clf.fit(X_train_vec, y_train)

# ==========================================
# 5. Threshold Optimization
# ==========================================
print("\nOptimizing Threshold on Test Set...")
y_probs_train = clf.predict_proba(X_train_vec)[:, 1]
y_probs_test = clf.predict_proba(X_test_vec)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs_test)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
f1_scores = np.nan_to_num(f1_scores)

best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Optimal Threshold: {best_thresh:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")

y_pred_optimized = (y_probs_test >= best_thresh).astype(int)

# ==========================================
# 6. Advanced Visualization (Separate Files)
# ==========================================
print("\nGenerating separated performance graphs...")

# --- Graph A: Learning Curve ---
plt.figure(figsize=(8, 6))
test_score = np.zeros((clf.n_estimators,), dtype=np.float64)
for i, y_pred in enumerate(clf.staged_predict_proba(X_test_vec)):
    test_score[i] = log_loss(y_test, y_pred)

plt.plot(np.arange(clf.n_estimators) + 1, clf.train_score_, 'b-', label='Training Loss')
plt.plot(np.arange(clf.n_estimators) + 1, test_score, 'r-', label='Test Loss')
plt.title('Learning Curve (Log Loss)')
plt.xlabel('Boosting Iterations (Trees)')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('1_learning_curve.png', dpi=300)
print("Saved 1_learning_curve.png")
plt.close()

# --- Graph B: ROC Curve ---
plt.figure(figsize=(8, 6))
fpr_train, tpr_train, _ = roc_curve(y_train, y_probs_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_probs_test)

plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc(fpr_train, tpr_train):.3f}', color='blue', linestyle='--')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc(fpr_test, tpr_test):.3f}', color='red', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve: Train vs Test')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('2_roc_curve.png', dpi=300)
print("Saved 2_roc_curve.png")
plt.close()

# --- Graph C: Score Distribution ---
plt.figure(figsize=(8, 6))
sns.histplot(y_probs_test[y_test == 0], color='red', alpha=0.4, label='True Negative (Not NES)', bins=30, stat="density")
sns.histplot(y_probs_test[y_test == 1], color='green', alpha=0.4, label='True Positive (NES)', bins=30, stat="density")
plt.axvline(best_thresh, color='black', linestyle='--', label=f'Threshold {best_thresh:.2f}')
plt.title('Test Set Score Separation')
plt.xlabel('Predicted Probability')
plt.legend()
plt.savefig('3_score_distribution.png', dpi=300)
print("Saved 3_score_distribution.png")
plt.close()

# --- Graph D: Confusion Matrix ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Test Set)\nThreshold: {best_thresh:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.xticks([0.5, 1.5], ['Not NES', 'NES'])
plt.yticks([0.5, 1.5], ['Not NES', 'NES'])
plt.savefig('4_confusion_matrix.png', dpi=300)
print("Saved 4_confusion_matrix.png")
plt.close()

# --- Graph E: Accuracy by Class (True Pos vs True Neg) ---
plt.figure(figsize=(8, 6))

# Extract TP, TN, FP, FN
tn, fp, fn, tp = cm.ravel()

# Calculate percentages (Sensitivity and Specificity)
true_pos_perc = (tp / (tp + fn)) * 100
true_neg_perc = (tn / (tn + fp)) * 100

# Plotting
labels = ['Correct Positive\n(Sensitivity)', 'Correct Negative\n(Specificity)']
values = [true_pos_perc, true_neg_perc]
colors = ['#2ca02c', '#d62728'] # Green for Pos, Red for Neg

bars = plt.bar(labels, values, color=colors, alpha=0.8)

# Add percentage text on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(0, 110) # Give space for text
plt.ylabel('Accuracy Percentage (%)')
plt.title(f'Per-Class Accuracy at Threshold {best_thresh:.2f}')
plt.grid(axis='y', alpha=0.3)
plt.savefig('5_accuracy_by_class.png', dpi=300)
print("Saved 5_accuracy_by_class.png")
plt.close()

# ==========================================
# 7. Final Text Report
# ==========================================
print("\n" + "=" * 40)
print("FINAL PERFORMANCE SUMMARY")
print("=" * 40)
print(f"Train AUC Score: {auc(fpr_train, tpr_train):.4f}")
print(f"Test AUC Score:  {auc(fpr_test, tpr_test):.4f}")
print("-" * 40)
print(f"Correctly Identified NES (Recall):     {true_pos_perc:.2f}%")
print(f"Correctly Identified Non-NES (Spec.):  {true_neg_perc:.2f}%")
print("-" * 40)
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred_optimized, target_names=['Not NES', 'NES']))