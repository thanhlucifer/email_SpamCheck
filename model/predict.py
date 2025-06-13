import joblib
from preprocessing.text_cleaner import clean_text

def predict_email(text):
    # Load model và vectorizer
    model = joblib.load('model/spam_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')

    # Làm sạch văn bản
    cleaned = clean_text(text)

    # Vector hóa văn bản
    X = vectorizer.transform([cleaned])

    # Dự đoán
    prediction = model.predict(X)[0]
    return "Spam" if prediction == 1 else "Ham"
