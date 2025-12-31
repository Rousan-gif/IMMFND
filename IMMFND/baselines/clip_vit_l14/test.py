import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import AutoProcessor, CLIPModel
from train_clip_vit_l14 import FakeNewsDataset, load_dataset, BASE_PATH, BATCH_SIZE, MAX_TEXT_LENGTH
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Load saved model + processor
processor = AutoProcessor.from_pretrained("./clip_vit_l14_trained")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)


class CLIPForFakeNewsEval(torch.nn.Module):
    def __init__(self, clip_model, combined_dim):
        super().__init__()
        self.clip = clip_model
        self.classifier = torch.nn.Linear(combined_dim, 2)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        image_features = self.clip.get_image_features(
            pixel_values=pixel_values
        )

        combined = torch.cat([text_features, image_features], dim=1)
        logits = self.classifier(combined)

        return logits


print("Loading test dataset...")
test_df = load_dataset("test")
test_dataset = FakeNewsDataset(test_df, processor)

# Find combined feature dimension
dummy = next(iter(DataLoader(test_dataset, batch_size=1)))
dummy = {k: v.to(DEVICE) for k,v in dummy.items()}

text_feat  = clip_model.get_text_features(dummy["input_ids"], dummy["attention_mask"])
image_feat = clip_model.get_image_features(dummy["pixel_values"])

combined_dim = text_feat.shape[1] + image_feat.shape[1]

model = CLIPForFakeNewsEval(clip_model, combined_dim).to(DEVICE)
model.load_state_dict(torch.load("./clip_vit_l14_trained/pytorch_model.bin"))
model.eval()


def evaluate(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k,v in batch.items()}

            logits = model(**batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_map = ["fake", "real"]

    print("\n📊 Test Classification Report:\n")
    print(classification_report(
        [label_map[l] for l in all_labels],
        [label_map[p] for p in all_preds],
        target_names=label_map
    ))


evaluate(test_dataset)
