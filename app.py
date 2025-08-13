import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence:** {pred:.2f}")
