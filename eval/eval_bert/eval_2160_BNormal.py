import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import type_of_target
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

# === Load fine-tuned model ===
model_path = "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_2160_normal"  # CHANGE THIS
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# === Load test data ===
test_data = pd.read_csv("C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_2160_test_normal.csv")  # expects 'text' and 'label' columns
texts = test_data['text'].tolist()
y_true = test_data['label'].tolist()

# === Setup device ===
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'cuda:0' if device == 0 else 'CPU'}")

# === Pipeline ===
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, top_k=1)

# === Predict ===
y_pred = []
for text in texts:
    result = pipeline(text)[0][0]  # Fix: double indexing
    label = result['label']
    try:
        y_pred.append(int(label.replace("LABEL_", "")))
    except:
        y_pred.append(1 if "POS" in label.upper() else 0)


# === Evaluate ===
accuracy = accuracy_score(y_true, y_pred)

# Auto-detect binary/multiclass for F1 and confusion matrix unpacking
problem_type = type_of_target(y_true)
if problem_type == "binary":
    f1 = f1_score(y_true, y_pred, average="binary")
else:
    f1 = f1_score(y_true, y_pred, average="weighted")

# Print core metrics
print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# Only unpack TN/FP if binary
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
else:
    print("TN/FP not available — this is not a binary classification task.")

# === Plot confusion matrix ===
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Pred {i}' for i in range(cm.shape[0])],
            yticklabels=[f'True {i}' for i in range(cm.shape[0])])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
