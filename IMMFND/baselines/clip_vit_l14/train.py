import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, CLIPModel, TrainingArguments, Trainer
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
BASE_PATH = "/scratch/sg/Rousan/Finla_Dataset_20_10_Text_only_correct"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_TEXT_LENGTH = 77

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -----------------------------
# LOAD DATA HELPERS
# -----------------------------
def load_dataset(split):
    data = []
    split_path = os.path.join(BASE_PATH, split)

    for label_name, image_folder in [("fake", "Fake_image"), ("real", "Real_image")]:
        xl_folder = os.path.join(split_path, label_name.capitalize())
        image_dir = os.path.join(xl_folder, image_folder)

        for xl_file in os.listdir(xl_folder):
            if xl_file.endswith(".xlsx"):
                df = pd.read_excel(os.path.join(xl_folder, xl_file))

                for _, row in df.iterrows():
                    sr_no = str(row["Sr. No"])
                    image_name = sr_no + ".jpg"
                    image_path = os.path.join(image_dir, image_name)

                    data.append({
                        "sr_no": sr_no,
                        "claim": row.get("claim", ""),
                        "label": label_name,
                        "image_path": image_path
                    })

    return pd.DataFrame(data)


print("Loading datasets...")
train_df = load_dataset("train")
val_df   = load_dataset("validation")
test_df  = load_dataset("test")

print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")


# -----------------------------
# LOAD CLIP
# -----------------------------
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)


# -----------------------------
# DATASET CLASS
# -----------------------------
class FakeNewsDataset(Dataset):
    def __init__(self, df, processor, max_length=MAX_TEXT_LENGTH):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length
        self.label_to_id = {"fake": 0, "real": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except:
            image = Image.new("RGB", (224,224), "white")

        text = str(row.get("claim",""))

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        inputs = {k: v.squeeze(0) for k,v in inputs.items()}
        inputs["labels"] = torch.tensor(self.label_to_id[row["label"]])

        return inputs


train_dataset = FakeNewsDataset(train_df, processor)
val_dataset   = FakeNewsDataset(val_df, processor)
test_dataset  = FakeNewsDataset(test_df, processor)


# -----------------------------
# MODEL WRAPPER
# -----------------------------
class CLIPForFakeNews(nn.Module):
    def __init__(self, clip_model, combined_dim):
        super().__init__()
        self.clip = clip_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(combined_dim, 2)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        image_features = self.clip.get_image_features(
            pixel_values=pixel_values
        )

        combined = torch.cat([text_features, image_features], dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# -----------------------------
# DETERMINE FEATURE DIM
# -----------------------------
dummy = next(iter(DataLoader(train_dataset, batch_size=1)))
dummy = {k: v.to(DEVICE) for k,v in dummy.items()}

text_feat  = clip_model.get_text_features(
    input_ids=dummy["input_ids"],
    attention_mask=dummy["attention_mask"]
)

image_feat = clip_model.get_image_features(
    pixel_values=dummy["pixel_values"]
)

combined_dim = text_feat.shape[1] + image_feat.shape[1]

model = CLIPForFakeNews(clip_model, combined_dim).to(DEVICE)


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -----------------------------
# TRAINING ARGS
# -----------------------------
training_args = TrainingArguments(
    output_dir="./clip_vit_l14_results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)


# -----------------------------
# TRAINER
# -----------------------------
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# -----------------------------
# TRAIN
# -----------------------------
print("\n Training CLIP-ViT-L/14 Fake News Model...\n")
trainer.train()

print("\n Training complete. Saving model...\n")
trainer.save_model("./clip_vit_l14_trained")
processor.save_pretrained("./clip_vit_l14_trained")

print(" Saved to: ./clip_vit_l14_trained")

