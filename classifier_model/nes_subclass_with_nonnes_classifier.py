import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Import the feature extraction logic from your existing file
from nes_feature_extractor import process_dataframe

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
POS_CSV_PATH = r"data\nes_sequences_for_model.csv"
NEG_CSV_PATH =  r"data\negatives_dataset.csv"
MODEL_SAVE_PATH = "nes_subclass_with_nonnes_classifier_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MODEL DEFINITION ---

class NESUniversalClassifier(nn.Module):
    """
    MLP Classifier designed to distinguish between NES subclasses and Non-NES sequences.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- BINARY EVALUATION LOGIC ---

def evaluate_binary_performance(all_true_indices, all_preds_indices, class_names, set_name):
    """
    Collapses multi-class results into a binary NES vs Non-NES evaluation.
    """
    # Identify which index belongs to 'Non-NES'
    try:
        non_nes_idx = class_names.index('Non-NES')
    except ValueError:
        print("Error: 'Non-NES' class not found in class_names.")
        return

    # Convert indices to binary: 0 for Non-NES, 1 for any NES subclass
    binary_true = [0 if i == non_nes_idx else 1 for i in all_true_indices]
    binary_preds = [0 if i == non_nes_idx else 1 for i in all_preds_indices]

    print(f"\n" + "#"*50)
    print(f"   BINARY EVALUATION (NES vs Non-NES): {set_name}")
    print("#"*50)
    
    print(classification_report(binary_true, binary_preds, target_names=['Non-NES', 'NES']))

    # Binary Confusion Matrix
    cm_bin = confusion_matrix(binary_true, binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Predicted Non-NES', 'Predicted NES'], 
                yticklabels=['Actual Non-NES', 'Actual NES'])
    plt.title(f'Binary Confusion Matrix - {set_name}')
    plt.show()


# --- UTILITY FUNCTIONS ---

def add_faiss_features(X_train, X_val, X_test, y_train, num_classes):
    """
    Enriches features by calculating Euclidean distance to each class centroid.
    Ensures consistent dimensions across Train, Val, and Test.
    """
    d = X_train.shape[1]
    centroids = []
    for i in range(num_classes):
        class_data = X_train[y_train == i]
        if len(class_data) > 0:
            centroids.append(class_data.mean(axis=0))
        else:
            centroids.append(np.zeros(d))
            
    centroids = np.array(centroids).astype('float32')
    
    def get_dists(data):
        data_float = data.astype('float32')
        all_dists = []
        for c in centroids:
            d_to_c = np.linalg.norm(data_float - c, axis=1)
            all_dists.append(d_to_c)
        return np.column_stack(all_dists)

    return np.hstack([X_train, get_dists(X_train)]), \
           np.hstack([X_val,   get_dists(X_val)]), \
           np.hstack([X_test,  get_dists(X_test)])

def evaluate_set(model, loader, class_names, set_name):
    """
    Evaluates the model on a specific set and prints a labeled report + Confusion Matrix.
    """
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.numpy())
    
    print(f"\n" + "="*50)
    print(f"   FINAL EVALUATION: {set_name}")
    print("="*50)
    
    all_indices = list(range(len(class_names)))
    print(classification_report(all_true, all_preds, labels=all_indices, 
                                target_names=class_names, zero_division=0))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(all_true, all_preds, labels=all_indices)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {set_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()
    
    # Call Binary Evaluation
    evaluate_binary_performance(all_true, all_preds, class_names, set_name)

    return all_true, all_preds

def run_universal_training_pipeline():
    # 1. Load Positive and Negative Data
    print("Loading datasets...")
    df_pos = pd.read_csv(POS_CSV_PATH)
    df_neg = pd.read_csv(NEG_CSV_PATH)
    
    # Clean NES labels
    df_pos["merged_class"] = df_pos["class"].str.extract(r"^(.+?)-")
    df_pos["merged_class"].fillna(df_pos["class"], inplace=True)
    df_pos = df_pos[['aa_sequence', 'merged_class']].copy()
    df_pos.columns = ['aa_sequence', 'label']

    # Sample Negatives (Non-NES)
    df_neg = df_neg[['sequence']].rename(columns={'sequence': 'aa_sequence'})
    num_neg_to_sample = len(df_pos) # Balance the dataset 1:1 ratio
    df_neg_sampled = df_neg.sample(n=num_neg_to_sample, random_state=42).copy()
    df_neg_sampled['label'] = 'Non-NES'

    # Combine datasets
    df_combined = pd.concat([df_pos, df_neg_sampled], axis=0).reset_index(drop=True)
    
    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(df_combined['label'])
    class_names = list(le.classes_)
    num_classes = len(class_names)
    print(f"Dataset combined. Total samples: {len(df_combined)}")
    print(f"Classes: {class_names}")

    # 2. Feature Extraction
    print("Calculating physical features for combined dataset...")
    X = process_dataframe(df_combined, col_name="aa_sequence")

    # 3. Data Splitting (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 4. Feature Enrichment with FAISS-like Centroids
    print("Enriching features with centroid distances...")
    X_train, X_val, X_test = add_faiss_features(X_train, X_val, X_test, y_train, num_classes)
    
    # 5. Scaling and Dataloaders
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 6. Initialize Model & Training
    model = NESUniversalClassifier(X_train.shape[1], 64, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_acc_hist, val_acc_hist = [], []

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        t_correct, t_total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(outputs, 1)
            t_total += yb.size(0)
            t_correct += (pred == yb).sum().item()
        
        # Epoch Validation Check
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(DEVICE))
                _, pred = torch.max(out, 1)
                v_total += yb.size(0)
                v_correct += (pred == yb).sum().item()

        train_acc = 100 * t_correct / t_total
        val_acc = 100 * v_correct / v_total
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # 7. Final Evaluations (VAL and TEST)
    evaluate_set(model, val_loader, class_names, "VALIDATION SET (VAL)")
    evaluate_set(model, test_loader, class_names, "TEST SET")

    # 8. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Universal Detector saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_universal_training_pipeline()