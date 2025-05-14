import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import type_of_target
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TextClassificationPipeline


print("=== Load fine-tuned RoBERTa model ===")
model_path = "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_4160_normal"  
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)


print("=== Load test data ===")
test_data = pd.read_csv("C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_4160_test_normal.csv")
texts = test_data['text'].tolist()
y_true = test_data['label'].tolist()

print("=== Setup device ===")
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'cuda:0' if device == 0 else 'CPU'}")

print("=== Pipeline ===")
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, top_k=1)

print("=== Predict ===")
y_pred = []
for text in texts:
    result = pipeline(text)[0][0]
    label = result['label']
    try:
        y_pred.append(int(label.replace("LABEL_", "")))
    except:
        y_pred.append(1 if "POS" in label.upper() else 0)

print("=== Evaluate ===")
accuracy = accuracy_score(y_true, y_pred)


problem_type = type_of_target(y_true)
if problem_type == "binary":
    f1 = f1_score(y_true, y_pred, average="binary")
else:
    f1 = f1_score(y_true, y_pred, average="weighted")



print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")


cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)



sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Pred {i}' for i in range(cm.shape[0])],
            yticklabels=[f'True {i}' for i in range(cm.shape[0])])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
