import pandas as pd
from textblob import TextBlob


def get_sentiment_label(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


# Load your CSV
Input_file = 'input.csv'
# Change this to your actual file
df = pd.read_csv(Input_file)

# Analyze sentiment
df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.
                                  polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.
                                      subjectivity)
df['sentiment_label'] = df['polarity'].apply(get_sentiment_label)

# Save to new CSV
output_file = 'output_sentiment.csv'
df.to_csv(output_file, index=False)

print(f"Sentiment analysis complete! Results saved to '{output_file}'")
