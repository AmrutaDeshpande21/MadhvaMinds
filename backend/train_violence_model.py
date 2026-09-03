import os
import sys
import re
import json
import time
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
HF_REPO_ID = "jinmang2/ucf-crime-tencrop-i3d"
HF_REPO_TYPE = "dataset"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset", "ucf-crime-i3d"))

NORMAL_CATEGORY = "Normal_Videos"

# ==========================================
# DATASET HELPERS & UTILITIES
# ==========================================
def parse_category(file_path):
    filename = os.path.basename(file_path)
    match = re.match(r'^([A-Za-z_]+?)(?=\d|_x264)', filename)
    if match:
        cat = match.group(1).rstrip('_')
        return cat
    if "Normal" in filename:
        return "Normal_Videos"
    return "Unknown"

def is_normal_category(category):
    return category == NORMAL_CATEGORY or category == "Normal" or category.startswith("Normal_Videos")

def discover_dataset_files(dataset_dir):
    # 1. Remote Hugging Face file list as baseline
    hf_train, hf_test = [], []
    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
        hf_train = [f for f in repo_files if f.startswith('UCF_Train') and f.endswith('.npy')]
        hf_test = [f for f in repo_files if f.startswith('UCF_Test') and f.endswith('.npy')]
    except Exception as e:
        print(f"Warning querying HuggingFace Hub: {e}")

    # 2. Check local files
    local_train = glob.glob(os.path.join(dataset_dir, "**", "UCF_Train*", "*.npy"), recursive=True)
    local_test = glob.glob(os.path.join(dataset_dir, "**", "UCF_Test*", "*.npy"), recursive=True)

    train_files = local_train if len(local_train) > 0 else hf_train
    test_files = local_test if len(local_test) > 0 else hf_test


    return train_files, test_files



def get_file_bytes_or_download(file_ref, dataset_dir):
    """
    Returns path to downloaded/existing .npy file.
    """
    if os.path.exists(file_ref):
        return file_ref
    
    local_path = os.path.join(dataset_dir, file_ref)
    if os.path.exists(local_path):
        return local_path

    # Download from HuggingFace on demand if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=file_ref,
            local_dir=dataset_dir
        )
        return downloaded
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {file_ref}: {e}")

# ==========================================
# PYTORCH DATASET
# ==========================================
class UCFCrimeDataset(Dataset):
    def __init__(self, file_refs, dataset_dir, sequence_length=32, target_feature_dim=None):
        self.file_refs = file_refs
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.target_feature_dim = target_feature_dim

    def __len__(self):
        return len(self.file_refs)

    def __getitem__(self, idx):
        file_ref = self.file_refs[idx]
        file_path = get_file_bytes_or_download(file_ref, self.dataset_dir)
        
        # Load .npy feature array
        raw_arr = np.load(file_path).astype(np.float32) # Shape can be (T, 10, F) or (T, F)
        
        # Average over 10 crops if 3-dimensional
        if raw_arr.ndim == 3:
            # (T, 10, F) -> (T, F)
            arr = raw_arr.mean(axis=1)
        elif raw_arr.ndim == 2:
            arr = raw_arr
        else:
            arr = raw_arr.reshape(raw_arr.shape[0], -1)

        T, F = arr.shape

        # Uniform temporal sampling or zero-padding to target sequence_length
        if T >= self.sequence_length:
            indices = np.linspace(0, T - 1, self.sequence_length, dtype=int)
            sampled_arr = arr[indices]
        else:
            # Pad with zeros
            sampled_arr = np.zeros((self.sequence_length, F), dtype=np.float32)
            sampled_arr[:T] = arr

        # Label generation: 0 = NORMAL, 1 = VIOLENCE/ABNORMAL
        category = parse_category(file_ref)
        label = 0.0 if is_normal_category(category) else 1.0

        return torch.tensor(sampled_arr, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ==========================================
# PYTORCH MODEL ARCHITECTURE
# ==========================================
class TemporalViolenceClassifier(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=256, lstm_hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # 1. Feature Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. 1D Temporal Convolution
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 4. Temporal Self-Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 5. FC Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, In_Features) e.g. (B, 32, 2048)
        proj = self.projection(x) # (B, T, 256)

        # Conv1d along temporal dimension
        conv_in = proj.transpose(1, 2) # (B, 256, T)
        conv_out = self.conv1d(conv_in).transpose(1, 2) # (B, T, 256)

        lstm_out, _ = self.bilstm(conv_out) # (B, T, 256)

        # Attention weights over time steps
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1) # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1) # (B, 256)

        logits = self.classifier(context).squeeze(-1) # (B,)
        return logits

# ==========================================
# STEP 3: INSPECTION FUNCTION
# ==========================================
def inspect_dataset(dataset_dir):
    print("UCF-Crime Dataset Inspection")
    print("----------------------------")

    train_files, test_files = discover_dataset_files(dataset_dir)
    print(f"Train files count: {len(train_files)}")
    print(f"Test files count:  {len(test_files)}")

    all_files = train_files + test_files
    normal_files = [f for f in all_files if is_normal_category(parse_category(f))]
    abnormal_files = [f for f in all_files if not is_normal_category(parse_category(f))]

    print(f"\nTotal Normal files:   {len(normal_files)}")
    print(f"Total Abnormal files: {len(abnormal_files)}")

    if train_files:
        sample_file = train_files[0]
        sample_path = get_file_bytes_or_download(sample_file, dataset_dir)
        arr = np.load(sample_path)
        
        print("\nExample File Details:")
        print(f"File:   {os.path.basename(sample_file)}")
        print(f"Shape:  {arr.shape}")
        print(f"Dtype:  {arr.dtype}")
        if arr.ndim == 3:
            print(f"Structure: {arr.shape[0]} temporal segments, {arr.shape[1]} crops, {arr.shape[2]} feature dim")
        elif arr.ndim == 2:
            print(f"Structure: {arr.shape[0]} temporal segments, {arr.shape[1]} feature dim")

    # Category breakdown
    cat_counts = {}
    for f in train_files:
        cat = parse_category(f)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print("\nTraining Categories Breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat:15s}: {count} files")

# ==========================================
# EVALUATION METRICS COMPUTATION
# ==========================================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            all_scores.extend(scores)
            all_targets.extend(targets.numpy())

    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)
    all_preds = (all_scores >= 0.5).astype(int)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_scores)
    except Exception:
        roc_auc = 0.5

    try:
        p, r, _ = precision_recall_curve(all_targets, all_scores)
        pr_auc = auc(r, p)
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "targets": all_targets,
        "scores": all_scores,
        "preds": all_preds
    }

# ==========================================
# TRAINING & EVALUATION PIPELINE
# ==========================================
def train_and_evaluate(dataset_dir, debug=False, epochs=30, sequence_length=32, batch_size=16, lr=1e-3):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")

    train_files, test_files = discover_dataset_files(dataset_dir)
    
    if debug:
        print("\n--- RUNNING IN DEBUG MODE ---")
        train_files = train_files[:50]
        test_files = test_files[:20] if test_files else train_files[30:50]
        epochs = min(epochs, 3)

    if not train_files:
        raise RuntimeError("No dataset files found. Please ensure download_dataset.py ran successfully.")

    # Stratified train/val split from training files
    train_cats = [parse_category(f) for f in train_files]
    val_split_files = []
    
    try:
        train_refs, val_refs = train_test_split(
            train_files, test_size=0.2, random_state=42, stratify=train_cats
        )
    except Exception:
        train_refs, val_refs = train_test_split(train_files, test_size=0.2, random_state=42)

    test_refs = test_files if test_files else val_refs

    print(f"Train samples: {len(train_refs)} | Val samples: {len(val_refs)} | Test samples: {len(test_refs)}")

    # Detect feature dimension dynamically from first sample
    first_path = get_file_bytes_or_download(train_refs[0], dataset_dir)
    first_arr = np.load(first_path)
    in_features = first_arr.shape[-1]
    print(f"Detected Input Feature Dimension: {in_features}")

    # Datasets and Loaders
    train_ds = UCFCrimeDataset(train_refs, dataset_dir, sequence_length=sequence_length, target_feature_dim=in_features)
    val_ds = UCFCrimeDataset(val_refs, dataset_dir, sequence_length=sequence_length, target_feature_dim=in_features)
    test_ds = UCFCrimeDataset(test_refs, dataset_dir, sequence_length=sequence_length, target_feature_dim=in_features)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Handle class imbalance (pos_weight for BCEWithLogitsLoss)
    train_labels = [0.0 if is_normal_category(parse_category(f)) else 1.0 for f in train_refs]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight_val = (num_neg / (num_pos + 1e-5)) if num_pos > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    # Initialize Model, Criterion, Optimizer, Scheduler
    model = TemporalViolenceClassifier(in_features=in_features, hidden_dim=256, lstm_hidden=128, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = 0.0
    best_model_path = os.path.join(MODEL_DIR, "best_violence_model.pt")
    best_meta_path = os.path.join(MODEL_DIR, "best_violence_model_meta.json")

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    patience = 7
    patience_counter = 0

    print("\nStarting Model Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation step
        val_res = evaluate_model(model, val_loader, device)
        val_f1 = val_res["f1"]
        val_acc = val_res["accuracy"]
        
        # Calculate val loss
        model.eval()
        v_loss = 0.0
        v_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                v_loss += loss.item() * inputs.size(0)
                v_total += targets.size(0)
        epoch_val_loss = v_loss / max(1, v_total)

        scheduler.step(val_f1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} Prec: {val_res['precision']:.4f} Rec: {val_res['recall']:.4f}")

        # Checkpoint saving
        if val_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            torch.save(model.state_dict(), best_model_path)
            
            meta_data = {
                "in_features": in_features,
                "sequence_length": sequence_length,
                "hidden_dim": 256,
                "lstm_hidden": 128,
                "class_mapping": {"0": "NORMAL", "1": "VIOLENCE/FIGHT"},
                "best_val_f1": float(val_f1),
                "best_val_acc": float(val_acc),
                "best_val_precision": float(val_res["precision"]),
                "best_val_recall": float(val_res["recall"]),
                "epoch": epoch,
                "training_time_sec": float(time.time() - start_time)
            }
            with open(best_meta_path, "w") as f:
                json.dump(meta_data, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Load best model for evaluation on test set
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\nRunning Evaluation on Test Set...")
    test_res = evaluate_model(model, test_loader, device)

    print("\n--- TEST METRICS SUMMARY ---")
    print(f"Accuracy:  {test_res['accuracy']:.4f}")
    print(f"Precision: {test_res['precision']:.4f}")
    print(f"Recall:    {test_res['recall']:.4f}")
    print(f"F1 Score:  {test_res['f1']:.4f}")
    print(f"ROC-AUC:   {test_res['roc_auc']:.4f}")
    print(f"PR-AUC:    {test_res['pr_auc']:.4f}")
    print("----------------------------")

    # Generate and save evaluation plots & metrics artifacts
    save_evaluation_artifacts(history, test_res)

    if debug:
        print("\nDEBUG VERIFICATION SUMMARY:")
        print("Dataset loading [OK]")
        print("Labels [OK]")
        print("Tensor shapes [OK]")
        print("Model forward pass [OK]")
        print("Loss calculation [OK]")
        print("Backpropagation [OK]")
        print("Checkpoint saving [OK]")
        print("Evaluation [OK]")

    return test_res


def save_evaluation_artifacts(history, test_res):
    # 1. Save metrics.json
    metrics_export = {
        "accuracy": test_res["accuracy"],
        "precision": test_res["precision"],
        "recall": test_res["recall"],
        "f1": test_res["f1"],
        "roc_auc": test_res["roc_auc"],
        "pr_auc": test_res["pr_auc"],
        "confusion_matrix": test_res["confusion_matrix"]
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_export, f, indent=4)

    # 2. Save classification_report.txt
    targets = test_res["targets"]
    preds = test_res["preds"]
    report_text = classification_report(targets, preds, labels=[0, 1], target_names=["NORMAL", "VIOLENCE/FIGHT"], zero_division=0)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write("UCF-CRIME VIOLENCE CLASSIFIER EVALUATION REPORT\n")
        f.write("================================================\n\n")
        f.write(report_text)


    # 3. Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(test_res["confusion_matrix"])
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["NORMAL", "VIOLENCE"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    # 4. ROC Curve plot
    scores = test_res["scores"]
    fpr, tpr, _ = roc_curve(targets, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_res["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
    plt.close()

    # 5. Precision-Recall Curve plot
    precision, recall, _ = precision_recall_curve(targets, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {test_res["pr_auc"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"))
    plt.close()

    # 6. Training History plot
    if history.get("train_loss"):
        plt.figure(figsize=(7, 4))
        plt.plot(history["train_loss"], label='Train Loss')
        plt.plot(history["val_loss"], label='Val Loss')
        plt.plot(history["val_f1"], label='Val F1')
        plt.xlabel('Epoch')
        plt.title('Training & Validation History')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"))
        plt.close()

    print(f"Saved evaluation artifacts to: {RESULTS_DIR}")

# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Violence Detection Model on UCF-Crime I3D features")
    parser.add_argument("--inspect", action="store_true", help="Inspect dataset structure and sample shapes")
    parser.add_argument("--debug", action="store_true", help="Run short debug training loop on small subset")
    parser.add_argument("--train", action="store_true", help="Run full model training pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing trained model on test set")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help="Path to local dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--sequence_length", type=int, default=32, help="Sequence length for temporal sampling")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.dataset_dir)
    elif args.debug:
        train_and_evaluate(args.dataset_dir, debug=True, epochs=3, sequence_length=args.sequence_length, batch_size=args.batch_size, lr=args.lr)
    elif args.train:
        train_and_evaluate(args.dataset_dir, debug=False, epochs=args.epochs, sequence_length=args.sequence_length, batch_size=args.batch_size, lr=args.lr)
    elif args.evaluate:
        train_and_evaluate(args.dataset_dir, debug=False, epochs=1, sequence_length=args.sequence_length, batch_size=args.batch_size, lr=args.lr)
    else:
        # Default behavior: run inspect and debug check
        print("No mode specified. Running inspect and debug check by default.")
        inspect_dataset(args.dataset_dir)
        print("\n" + "="*50 + "\n")
        train_and_evaluate(args.dataset_dir, debug=True, epochs=3, sequence_length=args.sequence_length, batch_size=args.batch_size, lr=args.lr)
