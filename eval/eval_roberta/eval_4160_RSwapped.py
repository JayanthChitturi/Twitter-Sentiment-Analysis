from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print('Load model and tokenizer')
model_path = "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_4160_swapped"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

print('Load swapped test set')
dataset = load_dataset("csv", data_files={"test": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_4160_test_swapped.csv"})
test_dataset = dataset["test"]

print('Tokenization')
def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

test_dataset = test_dataset.map(preprocess, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print('Load test set')
dataloader = DataLoader(test_dataset, batch_size=16)

print('Evaluate model')
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

print('Evaluation complete')
print("Classification Report (Swapped Test Set):")
print(classification_report(all_labels, all_preds))

print('Confusion Matrix (Swapped Test Set):')
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - RSwapped 4160 Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
