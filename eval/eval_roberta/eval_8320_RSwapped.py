from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and tokenizer
model_path = "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/test/model_swapped"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load normal test set
dataset = load_dataset("csv", data_files={"test": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_8320_test_swapped.csv"})
test_dataset = dataset["test"]

# Tokenization
def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

test_dataset = test_dataset.map(preprocess, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# DataLoader
dataloader = DataLoader(test_dataset, batch_size=16)

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch['label'].cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

# Metrics
print("Classification Report (Normal Test Set):")
print(classification_report(all_labels, all_preds))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Normal Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
