import os
import glob
import pandas as pd

# Project root directory
PROJECT_ROOT = r"C:/Users/HP/TAM/Twitter-Sentiment-Analysis"
DATA_DIR = os.path.join(PROJECT_ROOT, "data_final")
OUTPUT_CSV = os.path.join(DATA_DIR, "label_distribution_summary.csv")

def get_label_distribution(csv_path, label_type):
    """Compute label distribution for a CSV."""
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        return None
    
    # Total samples
    total = len(df)
    
    # Label counts
    label_counts = df['label'].value_counts(normalize=True).to_dict()
    
    # Map labels to sentiments based on label_type
    if label_type == "normal":
        # 0=negative, 1=positive, 2=neutral
        negative = label_counts.get(0, 0.0)
        positive = label_counts.get(1, 0.0)
        neutral = label_counts.get(2, 0.0)
    else:  # swapped
        # 0=positive, 1=negative, 2=neutral
        positive = label_counts.get(0, 0.0)
        negative = label_counts.get(1, 0.0)
        neutral = label_counts.get(2, 0.0)
    
    return {
        "negative": negative,
        "neutral": neutral,
        "positive": positive,
        "total_samples": total
    }

def main():
    """Compute label distributions for all train, test, dev CSVs."""
    os.chdir(PROJECT_ROOT)
    # Find all train, test, dev CSVs, exclude predictions
    pattern = os.path.join(DATA_DIR, "split_*_*.csv")
    csv_files = [f for f in glob.glob(pattern) 
                 if any(x in os.path.basename(f) for x in ["train", "test", "dev"])]
    
    if not csv_files:
        print("No train, test, or dev CSVs found in data/")
        return
    
    results = []
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        # Determine dataset type (train, test, dev)
        if "train" in file_name:
            dataset_type = "train"
        elif "test" in file_name:
            dataset_type = "test"
        elif "dev" in file_name:
            dataset_type = "dev"
        else:
            continue
        
        # Determine label type (normal or swapped)
        label_type = "swapped" if "swapped" in file_name else "normal"
        
        print(f"Processing {file_name} ({dataset_type}, {label_type})...")
        try:
            dist = get_label_distribution(csv_file, label_type)
            if dist:
                dist["file"] = file_name
                dist["type"] = dataset_type
                dist["label_type"] = label_type
                results.append(dist)
                print(f"Completed: {file_name}, Negative: {dist['negative']:.4f}, "
                      f"Neutral: {dist['neutral']:.4f}, Positive: {dist['positive']:.4f}, "
                      f"Total: {dist['total_samples']}")
            else:
                print(f"Skipping {file_name}: No 'label' column")
        except Exception as e:
            print(f"Failed to process {file_name}: {str(e)}")
    
    if results:
        # Save summary to CSV
        df = pd.DataFrame(results)
        df = df[["file", "type", "label_type", "negative", "neutral", "positive", "total_samples"]]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSummary saved to {OUTPUT_CSV}")
        # Display formatted table
        print("\nLabel Distribution Summary:")
        print(df[["file", "type", "label_type", "negative", "neutral", "positive"]].to_string(index=False))
    else:
        print("No distributions computed.")

if __name__ == "__main__":
    main()