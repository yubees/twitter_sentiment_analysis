import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("sentiment_lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Max length (same as training)
MAX_LEN = 100

# Text cleaning (basic for Twitter)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Predict function
def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded)[0][0]

    if pred > 0.6:
        return "Positive 😊"
    elif pred < 0.4:
        return "Negative 😡"
    else:
        return "Neutral 😐"

# Streamlit UI
st.title("🐦 Twitter Sentiment Analysis (LSTM)")

user_input = st.text_area("Enter Tweet")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: {result}")