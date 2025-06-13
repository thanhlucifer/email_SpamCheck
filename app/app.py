import streamlit as st
from model.predict import predict_spam

st.title("📧 Email Spam Detector")
email = st.text_area("Nhập nội dung email:")

if st.button("Kiểm tra"):
    if email.strip() == "":
        st.warning("Vui lòng nhập nội dung!")
    else:
        result = predict_spam(email)
        st.success(f"Kết quả: {result}")
