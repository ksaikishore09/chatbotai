# chatbot.py
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import pandas as pd
import random

# Download NLTK resources


nltk.download("punkt")
nltk.download("vader_lexicon")


# Load sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load and preprocess the dataset
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/ksaikishore09/modifieddataset/refs/heads/main/enhanced_chatbot_dataset_with_sentiment.csv"
    data = pd.read_csv(url)

    def format_sentence(sent):
        return {word: True for word in word_tokenize(sent.lower())}

    features = [(format_sentence(row["Message"]), row["Response"]) for _, row in data.iterrows()]
    random.shuffle(features)

    classifier = NaiveBayesClassifier.train(features)
    return classifier, format_sentence

classifier, format_sentence = train_model()

# Streamlit UI
st.title("🤖 Sentiment-Aware Chatbot")
st.write("Type a message and let the bot respond with sentiment awareness!")

user_input = st.text_input("You:")

if user_input:
    sentiment = sia.polarity_scores(user_input)
    compound = sentiment["compound"]

    if compound >= 0.05:
        label = "😊 Positive"
    elif compound <= -0.05:
        label = "😞 Negative"
    else:
        label = "😐 Neutral"

    response = classifier.classify(format_sentence(user_input))

    st.markdown(f"**🤖 Bot:** {response}")
    st.markdown(f"**Sentiment:** {label} (Score = {compound})")
