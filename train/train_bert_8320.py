import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch


# Load data
train = pd.read_csv('C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_8320_train_normal.csv')
dev = pd.read_csv('C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_8320_dev_normal.csv')
train_swapped = pd.read_csv('C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_8320_train_swapped.csv')


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)


# Custom Trainer with corrected compute_loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[unused-argument]
        labels = inputs.pop("labels").long()  # Convert to Long
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(model_name, output_dir, train_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # Convert to Dataset and tokenize
    train_ds = Dataset.from_pandas(train_data)
    dev_ds = Dataset.from_pandas(dev)
    # Sanitize labels: replace NaN/invalid with 0, ensure int
    train_ds = train_ds.map(lambda x: {'text': x['text'], 'label': 0 if x['label'] not in [0, 1, 2] else int(x['label'])})
    dev_ds = dev_ds.map(lambda x: {'text': x['text'], 'label': 0 if x['label'] not in [0, 1, 2] else int(x['label'])})
    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dev_ds = dev_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    # Rename and format
    train_ds = train_ds.rename_column('label', 'labels')
    dev_ds = dev_ds.rename_column('label', 'labels')
    train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    dev_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    # Debug
    print("First training sample:", train_ds[0])
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        fp16=True
    )
    trainer = CustomTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.train()
    trainer.save_model(output_dir)


# Train all four
train_model('bert-base-uncased', 'C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_8320_normal', train)
train_model('bert-base-uncased', 'C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_8320_swapped', train_swapped)
