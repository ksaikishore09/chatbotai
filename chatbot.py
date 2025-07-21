# chatbot.py
import streamlit as st
import nltk
import pandas as pd
import random
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import NaiveBayesClassifier, word_tokenize

# ✅ Download resources outside the cached function (RUNS ONCE)
nltk.download("punkt")
nltk.download("vader_lexicon")

# ✅ Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# ✅ Format function
def format_sentence(sent):
    return {word: True for word in word_tokenize(sent.lower())}

# ✅ Training the model (caches only data & model, not downloads)
@st.cache_resource
def train_model():
    data = pd.read_csv("https://raw.githubusercontent.com/ksaikishore09/modifieddataset/refs/heads/main/enhanced_chatbot_dataset_with_sentiment.csv")
    features = [(format_sentence(row["Message"]), row["Response"]) for _, row in data.iterrows()]
    random.shuffle(features)
    classifier = NaiveBayesClassifier.train(features)
    return classifier

classifier = train_model()

# ✅ Streamlit UI
st.title("🤖 Sentiment-Aware Chatbot")

user_input = st.text_input("You:")

if user_input:
    sentiment = sia.polarity_scores(user_input)
    compound = sentiment["compound"]

    if compound >= 0.05:
        sentiment_label = "😊 Positive"
    elif compound <= -0.05:
        sentiment_label = "😞 Negative"
    else:
        sentiment_label = "😐 Neutral"

    response = classifier.classify(format_sentence(user_input))

    st.markdown(f"**🤖 Bot:** {response}")
    st.markdown(f"**🧭 Sentiment:** {sentiment_label} (Score: {compound})")
