import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import os
import logging
import numpy as np
import gc


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


MODEL_PATHS = {
    "BERT Normal 2k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_2160_normal",
    "BERT Swapped 2k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_2160_swapped",
    "RoBERTa Normal 2k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_2160_normal",
    "RoBERTa Swapped 2k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_2160_swapped",
    "BERT Normal 4k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_4160_normal",
    "BERT Swapped 4k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_4160_swapped",
    "RoBERTa Normal 4k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_4160_normal",
    "RoBERTa Swapped 4k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_4160_swapped",
    "BERT Normal 8k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_8320_normal",
    "BERT Swapped 8k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/bert_8320_swapped",
    "RoBERTa Normal 8k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_8320_normal",
    "RoBERTa Swapped 8k": "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/models_final/roberta_8320_swapped",
}


METRICS = {
    "Model": [
        "BERT Normal 2k", "BERT Swapped 2k", "RoBERTa Normal 2k", "RoBERTa Swapped 2k",
        "BERT Normal 4k", "BERT Swapped 4k", "RoBERTa Normal 4k", "RoBERTa Swapped 4k",
        "BERT Normal 8k", "BERT Swapped 8k", "RoBERTa Normal 8k", "RoBERTa Swapped 8k",
    ],
    "Accuracy": [0.8287, 0.8056, 0.8426, 0.8380, 0.8534, 0.8630, 0.8606, 0.8606, 0.8329, 0.7728, 0.85, 0.85],
    "F1": [0.8219, 0.8015, 0.8402, 0.8352, 0.8552, 0.8646, 0.8613, 0.8613, 0.8190, 0.7815, 0.81, 0.81],
    "Eval Loss": [0.650, 0.650, 0.640, 0.630, 0.550, 0.550, 0.541, 0.531, 0.480, 0.480, 0.470, 0.460],
}



def load_pipeline(model_name, model_path):
    try:
        logger.info(f"Loading {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base" if "RoBERTa" in model_name else "bert-base-uncased")
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        logger.info(f"{model_name} loaded on {'GPU' if device == 0 else 'CPU'}")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return None


def plot_metrics(metric_type, group_by):
    df = pd.DataFrame(METRICS)
    plt.figure(figsize=(14, 6))
    if group_by == "Dataset Size":
        hue = "Model"
        x = [m.split()[-1] for m in df["Model"]]
    else:
        hue = [m.split()[-1] for m in df["Model"]]
        x = [m.split()[0] + " " + m.split()[1] for m in df["Model"]]
    
    sns.barplot(x=x, y=metric_type, hue=hue, data=df)
    plt.title(f"{metric_type} by {group_by}")
    plt.xlabel(group_by)
    plt.ylabel(metric_type)
    plt.xticks(rotation=45)
    plt.legend(title="Legend")
    plt.tight_layout()
    
    plot_path = "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/runs/metric_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def predict_sentiment(tweet):
    if not tweet:
        return "Please enter a tweet."
    results = []
    for model_name, model_path in MODEL_PATHS.items():
        with torch.no_grad():
            pipe = load_pipeline(model_name, model_path)
            if pipe:
                try:
                    pred = pipe(tweet)[0]
                    label = int(pred["label"].split("_")[1])
                    label_map = {0: "Positive", 1: "Negative", 2: "Neutral"} if "Swapped" in model_name else {0: "Negative", 1: "Positive", 2: "Neutral"}
                    results.append({
                        "Model": model_name,
                        "Prediction": label_map[label],
                        "Confidence": f"{pred['score']:.3f}"
                    })
                except Exception as e:
                    results.append({
                        "Model": model_name,
                        "Prediction": "Error",
                        "Confidence": str(e)
                    })
                del pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return pd.DataFrame(results).to_html()


def analyze_fine_tuning_resistance():
    df = pd.DataFrame(METRICS)
    
    
    variance = []
    model_types = ["BERT Normal", "BERT Swapped", "RoBERTa Normal", "RoBERTa Swapped"]
    for model_type in model_types:
        sub_df = df[df["Model"].str.contains(model_type)]
        var_acc = sub_df["Accuracy"].var()
        var_f1 = sub_df["F1"].var()
        var_loss = sub_df["Eval Loss"].var()
        variance.append({
            "Model Type": model_type,
            "Accuracy Variance": var_acc,
            "F1 Variance": var_f1,
            "Eval Loss Variance": var_loss
        })
    
    var_df = pd.DataFrame(variance)
    
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model Type", y="Accuracy Variance", data=var_df)
    plt.title("Accuracy Variance Across Dataset Sizes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    var_plot_path = "C:/Users/HP/TAM/Twitter-Sentiment-Analysis/runs/variance_plot.png"
    plt.savefig(var_plot_path)
    plt.close()
    
    
    most_stable = var_df.loc[var_df["Accuracy Variance"].idxmin()]["Model Type"]
    summary = f"**Fine-Tuning Resistance Analysis**\n\n"
    summary += "Resistance is measured by low variance in accuracy, F1, and eval_loss across dataset sizes (2k, 4k, 8k).\n"
    summary += f"- **Most Stable Model**: {most_stable} (accuracy variance: {var_df['Accuracy Variance'].min():.4f})\n"
    summary += "- **Variance Table**:\n" + var_df.to_string(index=False)
    return summary, var_plot_path


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Twitter Airline Sentiment Analysis Dashboard")
        
        with gr.Tab("Metrics Visualization"):
            metric_type = gr.Dropdown(choices=["Accuracy", "F1", "Eval Loss"], label="Metric Type", value="Accuracy")
            group_by = gr.Dropdown(choices=["Dataset Size", "Model Type"], label="Group By", value="Dataset Size")
            plot_button = gr.Button("Generate Plot")
            plot_output = gr.Image(label="Metric Plot")
            plot_button.click(
                fn=plot_metrics,
                inputs=[metric_type, group_by],
                outputs=plot_output
            )
        
        with gr.Tab("Sentiment Prediction"):
            tweet_input = gr.Textbox(label="Enter Tweet", placeholder="Type a tweet here...")
            predict_button = gr.Button("Predict Sentiment")
            prediction_output = gr.HTML(label="Predictions")
            predict_button.click(
                fn=predict_sentiment,
                inputs=[tweet_input],
                outputs=prediction_output
            )
        
        with gr.Tab("Fine-Tuning Resistance"):
            resistance_button = gr.Button("Analyze Resistance")
            resistance_summary = gr.Markdown()
            resistance_plot = gr.Image(label="Variance Plot")
            resistance_button.click(
                fn=analyze_fine_tuning_resistance,
                inputs=[],
                outputs=[resistance_summary, resistance_plot]
            )
    
    demo.launch(share=False)

if __name__ == "__main__":
    main()