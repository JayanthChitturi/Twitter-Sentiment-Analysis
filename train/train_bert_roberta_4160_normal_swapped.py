import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import os
from sklearn.metrics import accuracy_score, f1_score
import transformers

# Paths
data_dir = "data_final/"
model_dir = "models_final/"
os.makedirs(model_dir, exist_ok=True)

# Check Transformers version
print(f"Transformers version: {transformers.__version__}")
eval_key = "eval_strategy" if int(transformers.__version__.split(".")[0]) >= 4 and int(transformers.__version__.split(".")[1]) >= 41 else "evaluation_strategy"


# Custom Dataset
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# Training function
def train_model(model_type, model_name, train_file, dev_file, output_dir, label_type):
    # Load data
    try:
        train_df = pd.read_csv(train_file)
        dev_df = pd.read_csv(dev_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    
    # Validate dataset size
    expected_train_size = 3184  # ~80% of 3981
    expected_dev_size = 398    # ~10% of 3981
    if len(train_df) < expected_train_size * 0.9 or len(dev_df) < expected_dev_size * 0.9:
        print(f"Warning: Train size {len(train_df)} (expected ~{expected_train_size}), Dev size {len(dev_df)} (expected ~{expected_dev_size}). Check dataset.")
    
    # Initialize tokenizer and model
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    else:  # roberta
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Datasets
    train_dataset = TweetDataset(train_df, tokenizer)
    dev_dataset = TweetDataset(dev_df, tokenizer)
    
    # Training arguments
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,  # Effective batch size 16
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 10,
        eval_key: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "fp16": True,
        "dataloader_num_workers": 0,
        "max_grad_norm": 1.0
    }
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved {model_type} model to {output_dir}")


# Train all models
models = [
    ("bert", "bert-base-uncased", "normal"),
    ("bert", "bert-base-uncased", "swapped"),
    ("roberta", "roberta-base", "normal"),
    ("roberta", "roberta-base", "swapped")
]

for model_type, model_name, label_type in models:
    train_file = f"{data_dir}split_4160_train_{label_type}.csv"
    dev_file = f"{data_dir}split_4160_dev_{label_type}.csv"
    output_dir = f"{model_dir}{model_type}_4160_{label_type}"
    
    print(f"\nTraining {model_type} {label_type} model...")
    train_model(model_type, model_name, train_file, dev_file, output_dir, label_type)