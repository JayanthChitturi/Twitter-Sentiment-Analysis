# sentiment_model.py

from textblob import TextBlob


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    result = {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity,
        "sentiment_label": get_sentiment_label(sentiment.polarity)
    }
    return result


def get_sentiment_label(polarity):
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


# For quick testing
if __name__ == "__main__":
    text = input("Enter a tweet or text: ")
    print(analyze_sentiment(text))
