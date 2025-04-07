import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import matplotlib.animation as animation

# Set the title of the app
st.title("Twitter Sentiment Analysis 📊")

# Sidebar for instructions and additional features
st.sidebar.header("Instructions 📥")
st.sidebar.write("""
    1. Upload a CSV file containing tweets.
    2. View sentiment analysis of the tweets, including:
        - Polarity, Subjectivity, and Sentiment Labels (Positive, Negative, Neutral).
    3. View the animated word cloud generated from all the tweets.
    4. View sentiment trends over time (if the CSV includes a `date` column).
    5. Paste a tweet below for individual sentiment analysis.
""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file 📂", type=["csv"])

# Sentiment analysis function using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0:
        label = "Positive 😊"
    elif polarity < 0:
        label = "Negative 😞"
    else:
        label = "Neutral 😐"
    return polarity, subjectivity, label

# When the file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Ensure the dataframe has a 'text' column
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column with the tweet content. 😕")
    else:
        # Apply sentiment analysis to each tweet in the 'text' column
        df[['Polarity', 'Subjectivity', 'Sentiment Label']] = df['text'].apply(
            lambda x: pd.Series(analyze_sentiment(x))
        )

        # Show preview of the CSV with sentiment columns
        st.subheader("CSV Preview with Sentiment 📝")
        st.dataframe(df[['text', 'Polarity', 'Subjectivity', 'Sentiment Label']].head(5))

        # Button to view the full sentiment table
        if st.button("View Full Sentiment Table 📊"):
            st.subheader("Full Sentiment Table")
            st.dataframe(df[['text', 'Polarity', 'Subjectivity', 'Sentiment Label']])

        # Display overall sentiment breakdown in a bar chart with emojis
        st.subheader("Overall Sentiment Breakdown 📈")
        sentiment_counts = df['Sentiment Label'].value_counts()
        sentiment_counts = sentiment_counts.rename({
            "Positive 😊": "Positive",
            "Negative 😞": "Negative",
            "Neutral 😐": "Neutral"
        })
        st.bar_chart(sentiment_counts)

        # Animated Word Cloud Function
        def generate_word_cloud(text_data):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
            return wordcloud

        # Generate word cloud text from all tweets
        all_text = ' '.join(df['text'].astype(str).tolist())

        # Create animated word cloud using FuncAnimation
        fig, ax = plt.subplots(figsize=(8, 6))
        wordcloud = generate_word_cloud(all_text)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        def update(i):
            # Generate a new word cloud with modified parameters for each frame
            wordcloud = generate_word_cloud(all_text)
            ax.clear()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")

        # Animate the word cloud
        ani = animation.FuncAnimation(fig, update, frames=30, interval=500, repeat=True)

        # Display the animated word cloud
        st.subheader("Animated Word Cloud 🌪️")
        st.pyplot(fig)

        # Show sentiment trend over time (if a 'date' column exists)
        if 'date' in df.columns:
            st.subheader("Sentiment Trend Over Time 📅")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            trend_data = df.groupby(df['date'].dt.date)['Polarity'].mean()
            st.line_chart(trend_data)
        else:
            st.warning("No 'date' column found in the CSV for sentiment trend analysis. ⚠️")

# Add a text field where users can paste their tweet for analysis
st.subheader("Analyze a Single Tweet 📝")
tweet_input = st.text_area("Paste your tweet here for sentiment analysis:")

# Button to analyze the single tweet
if st.button("Analyze Tweet 📊"):
    if tweet_input.strip() != "":  # Check if tweet is not empty
        polarity, subjectivity, sentiment_label = analyze_sentiment(tweet_input)

        # Display the sentiment results for the single tweet
        st.write(f"Polarity: {polarity}")
        st.write(f"Subjectivity: {subjectivity}")
        st.write(f"Sentiment: {sentiment_label}")
    else:
        st.warning("Please enter a tweet to analyze.")
