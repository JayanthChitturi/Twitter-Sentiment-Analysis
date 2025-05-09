import pandas as pd
data = pd.read_csv('Twitter-Sentiment-Analysis/data/airline_sentiment_analysis.csv')
data = data[['text', 'airline_sentiment']].rename(columns={'airline_sentiment': 'label'})
data['label'] = data['label'].map({'positive': 1, 'negative': 0, 'neutral': 2})
data = data.sample(n=10000, random_state=42)
train = data.iloc[:8000]; dev = data.iloc[8000:9000]; test = data.iloc[9000:]
train.to_csv('Twitter-Sentiment-Analysis/data/train.csv', index=False)
dev.to_csv('Twitter-Sentiment-Analysis/data/dev.csv', index=False)
test.to_csv('Twitter-Sentiment-Analysis/data/test.csv', index=False)