import streamlit as st
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.text_cleaner import clean_text


# Load model và vectorizer
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Tiêu đề
st.title("📧 Email Spam Checker")

# Nhập email
text = st.text_area("Nhập nội dung email cần kiểm tra:")

if st.button("Kiểm tra"):
    if text.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung email.")
    else:
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        prediction = model.predict(X)[0]
        result = "🟥 SPAM" if prediction == 1 else "🟩 HAM (không phải spam)"
        st.success(f"Kết quả: {result}")
