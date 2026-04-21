import streamlit as st
import numpy as np
import pickle
import re
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # must match training

# Clean text (Twitter style)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Pad manually (no tensorflow needed)
def pad_sequence(seq, maxlen):
    padded = np.zeros((1, maxlen))
    seq = seq[:maxlen]
    padded[0, -len(seq):] = seq
    return padded.astype(np.int32)

# Prediction
def predict(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequence(seq, MAX_LEN)

    pred = session.run([output_name], {input_name: padded})[0][0][0]

    if pred > 0.6:
        sentiment = "Positive 😊"
    elif pred < 0.4:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"

    return sentiment, float(pred)

# UI
st.title("🐦 Twitter Sentiment Analysis (ONNX)")

text = st.text_area("Enter tweet")

if st.button("Analyze"):
    if text.strip():
        sentiment, score = predict(text)
        st.success(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {score:.2f}")
    else:
        st.warning("Please enter text")