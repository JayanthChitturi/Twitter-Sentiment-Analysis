import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Project root directory
PROJECT_ROOT = r"C:/Users/HP/TAM/Twitter-Sentiment-Analysis"
DATA_DIR = r"C:/Users/HP/TAM/Twitter-Sentiment-Analysis/data_final"
CSV_PATH = f"{DATA_DIR}/label_distribution_summary.csv"
OUTPUT_PATH = f"{DATA_DIR}/label_distribution_plot.png"

# Read the CSV
df = pd.read_csv(CSV_PATH)

# Create a column for split size (2160, 4160, 8320)
df['split_size'] = df['file'].str.extract(r'split_(\d+)_').astype(int)

# Create a display name for each dataset (e.g., "2160 Train Normal")
df['display_name'] = df.apply(lambda x: f"{x['split_size']} {x['type'].capitalize()} {x['label_type'].capitalize()}", axis=1)

# Sort by split_size, type, and label_type for consistent ordering
df = df.sort_values(['split_size', 'type', 'label_type'])

# Prepare data for plotting
labels = ['Negative', 'Neutral', 'Positive']
colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
datasets = df['display_name'].values
negative = df['negative'].values
neutral = df['neutral'].values
positive = df['positive'].values

# Set up the plot
plt.figure(figsize=(14, 8))
bar_width = 0.25
index = np.arange(len(datasets))

# Plot bars
plt.bar(index - bar_width, negative, bar_width, label='Negative', color=colors[0])
plt.bar(index, neutral, bar_width, label='Neutral', color=colors[1])
plt.bar(index + bar_width, positive, bar_width, label='Positive', color=colors[2])

# Customize the plot
plt.xlabel('Dataset')
plt.ylabel('Proportion')
plt.title('Label Distribution Across Train, Test, and Dev Datasets')
plt.xticks(index, datasets, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to {OUTPUT_PATH}")