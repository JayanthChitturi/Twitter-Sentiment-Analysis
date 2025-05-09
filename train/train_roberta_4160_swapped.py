import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'text': 'text', 'label': 'label'})  # Make sure these columns exist
    return Dataset.from_pandas(df)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def train_roberta(csv_path, model_save_path):
    print(f"\nTraining on Dataset: {csv_path}")
    
    # Load CSV as HuggingFace Dataset
    dataset = load_data(csv_path)
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    # Tokenizer & Model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)


    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=50,
        save_steps=500,
        do_train=True,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    torch.cuda.empty_cache()


    # Swapped dataset training (optional)
    train_roberta(
        "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final/split_4160_train_swapped.csv",
        "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_4160_swapped"
    )
