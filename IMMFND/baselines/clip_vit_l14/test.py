import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, CLIPModel, TrainingArguments, Trainer
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# METRICS FUNCTION
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# -----------------------------
# EVALUATION
# -----------------------------
def detailed_evaluation(model, dataset, dataset_name):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = torch.argmax(outputs["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    label_names = ["fake", "real"]
    pred_labels = [label_names[p] for p in all_preds]
    true_labels = [label_names[l] for l in all_labels]
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=label_names))
    return all_preds, all_labels

print("\nValidation Evaluation:")
val_preds, val_labels = detailed_evaluation(custom_model, val_dataset, "Validation")

print("\nTest Evaluation:")
test_preds, test_labels = detailed_evaluation(custom_model, test_dataset, "Test")
