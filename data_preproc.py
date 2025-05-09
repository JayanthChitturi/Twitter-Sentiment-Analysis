import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Input/output paths
input_path = "data/airline_sentiment_analysis.csv"
output_dir = "data_final/"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: File not found at {input_path}. Please check the path.")
    exit(1)

# Initial stats
print(f"Dataset size: {len(df)}")
print("\nSentiment distribution (%):")
print(df["airline_sentiment"].value_counts(normalize=True).round(3) * 100)

# Define scaled splits
original_sizes = [2160, 4160, 8320]
original_total = sum(original_sizes)  # 14640
scale_factor = min(1.0, len(df) / original_total)  # Cap at 1.0 if >14640
scaled_sizes = [int(size * scale_factor) for size in original_sizes]
scaled_total = sum(scaled_sizes)

# Adjust if scaled_total exceeds dataset size
while scaled_total > len(df):
    scaled_sizes[-1] -= 1  # Reduce largest split
    scaled_total = sum(scaled_sizes)

print(f"\nScaled split sizes: {scaled_sizes} (Total: {scaled_total})")

# Check if dataset is too small
if scaled_total < len(df) * 0.8:  # <80% of dataset
    print(f"Warning: Scaled total ({scaled_total}) uses <80% of dataset ({len(df)}). Consider supplementing data.")


# Label encoding
def encode_labels(df, swapped=False):
    if swapped:
        label_map = {"negative": 1, "positive": 0, "neutral": 2}
    else:
        label_map = {"negative": 0, "positive": 1, "neutral": 2}
    df["label"] = df["airline_sentiment"].map(label_map)
    return df[["text", "label"]]


# Generate splits
datasets = {}
for size in scaled_sizes:
    # Stratify by sentiment
    df_subset = df.groupby("airline_sentiment").apply(
        lambda x: x.sample(n=int(size * len(x) / len(df)), random_state=42)
    ).reset_index(drop=True)
    
    # Split into train (80%), dev (10%), test (10%)
    train, temp = train_test_split(df_subset, test_size=0.2, stratify=df_subset["airline_sentiment"], random_state=42)
    dev, test = train_test_split(temp, test_size=0.5, stratify=temp["airline_sentiment"], random_state=42)
    
    datasets[size] = {"train": train, "dev": dev, "test": test}
    
    # Save normal and swapped versions
    for split_name, split_df in datasets[size].items():
        # Normal labels
        df_normal = encode_labels(split_df.copy(), swapped=False)
        df_normal.to_csv(
            f"{output_dir}split_{size}_{split_name}_normal.csv", index=False
        )
        print(f"Saved split_{size}_{split_name}_normal.csv: {len(df_normal)} tweets")
        
        # Swapped labels
        df_swapped = encode_labels(split_df.copy(), swapped=True)
        df_swapped.to_csv(
            f"{output_dir}split_{size}_{split_name}_swapped.csv", index=False
        )
        print(f"Saved split_{size}_{split_name}_swapped.csv: {len(df_swapped)} tweets")
    
    # Print split stats
    print(f"\nSplit {size} sentiment distribution (%):")
    print(f"Train:\n{train['airline_sentiment'].value_counts(normalize=True).round(3) * 100}")
    print(f"Dev:\n{dev['airline_sentiment'].value_counts(normalize=True).round(3) * 100}")
    print(f"Test:\n{test['airline_sentiment'].value_counts(normalize=True).round(3) * 100}")