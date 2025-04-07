import pandas as pd
from textblob import TextBlob
import tkinter as tk
from tkinter import filedialog, messagebox
import os


def get_sentiment_label(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_csv(file_path):
    try:
        df = pd.read_csv(file_path)

        if 'text' not in df.columns:
            messagebox.showerror("Error", "CSV must have a 'text' column.")
            return

        df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        df['sentiment_label'] = df['polarity'].apply(get_sentiment_label)

        output_file = os.path.splitext(file_path)[0] + '_sentiment_output.csv'
        df.to_csv(output_file, index=False)

        messagebox.showinfo("Success", f"Analysis complete!\nSaved as:\n{output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


def upload_and_analyze():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        analyze_csv(file_path)


# Setup GUI
root = tk.Tk()
root.title("CSV Sentiment Analyzer")
root.geometry("300x150")

label = tk.Label(root, text="Upload a CSV file to analyze", pady=10)
label.pack()

upload_button = tk.Button(root, text="Upload CSV", command=upload_and_analyze)
upload_button.pack(pady=10)

root.mainloop()
# This script creates a simple GUI for uploading a CSV file and analyzing its sentiment.