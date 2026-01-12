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

# Import the feature extraction logic from your feature file
from nes_feature_extractor import process_dataframe

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
CSV_PATH =  r"data\nes_sequences_for_model.csv"
MODEL_SAVE_PATH = "nes_subclass_classifier_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MODEL DEFINITION ---

class NESClassifier(nn.Module):
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

# --- UTILITY FUNCTIONS ---

def add_faiss_features(X_train, X_val, X_test, y_train, num_classes):
    """
    Calculates distances to class centroids to enrich features.
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
    
    print(f"\n" + "="*45)
    print(f"   FINAL REPORT: {set_name}")
    print("="*45)
    
    all_indices = list(range(len(class_names)))
    print(classification_report(all_true, all_preds, labels=all_indices, 
                                target_names=class_names, zero_division=0))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(all_true, all_preds, labels=all_indices)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {set_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()
    
    return all_true, all_preds

def run_nes_training_pipeline():
    # 1. Load Data
    csv_path = CSV_PATH
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Preprocess classes: merge subclasses (e.g., '1a-1' -> '1a')
    df["merged_class"] = df["class"].str.extract(r"^(.+?)-")
    df["merged_class"].fillna(df["class"], inplace=True)
    
    # Encode labels to numerical values
    le = LabelEncoder()
    y = le.fit_transform(df["merged_class"])
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # 2. Extract physicochemical features
    X = process_dataframe(df)

    # 3. Data Splitting (80% Train, 10% Val, 10% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 4. Feature Enrichment: Add distances to class centroids (FAISS-like logic)
    X_train, X_val, X_test = add_faiss_features(X_train, X_val, X_test, y_train, num_classes)
    
    # 5. Scaling: Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch Tensors and create DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 6. Initialize Model, Optimizer, and Loss Function
    model = NESClassifier(X_train.shape[1], 64, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # History lists for plotting
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    # 7. Training Loop
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and accuracy stats
            t_loss += loss.item() * xb.size(0)
            _, pred = torch.max(outputs, 1)
            t_total += yb.size(0)
            t_correct += (pred == yb).sum().item()
        
        # Calculate average training metrics for the epoch
        avg_train_loss = t_loss / t_total
        train_acc = 100 * t_correct / t_total

        # --- Validation Phase ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                
                # Accumulate validation stats
                v_loss += loss.item() * xb.size(0)
                _, pred = torch.max(out, 1)
                v_total += yb.size(0)
                v_correct += (pred == yb).sum().item()

        # Calculate average validation metrics
        avg_val_loss = v_loss / v_total
        val_acc = 100 * v_correct / v_total
        
        # Store metrics in history
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:5.1f}% | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:5.1f}%")

    # 8. Visualization: Plot Accuracy and Loss
    plt.figure(figsize=(12, 5))

    # Plot Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_loss_hist, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), val_loss_hist, label='Val Loss', linestyle='--')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_acc_hist, label='Train Accuracy')
    plt.plot(range(1, EPOCHS + 1), val_acc_hist, label='Val Accuracy', linestyle='--')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_performance_curves.png")
    plt.show()

    # 9. Final Performance Evaluation on Test Set
    evaluate_set(model, test_loader, class_names, "TEST SET")

    # 10. Save trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_nes_training_pipeline()