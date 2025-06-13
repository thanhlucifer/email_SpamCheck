import joblib
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing.text_cleaner import clean_text

def predict_spam(input_text):
    model = joblib.load('model/spam_model.pkl')
    
    cleaned = clean_text(input_text)
    vectorizer = CountVectorizer()
    vectorizer.fit([cleaned])  # Fit mới vectorizer (có thể lưu vectorizer lại khi train tốt hơn)

    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)
    return 'Spam' if prediction[0] == 1 else 'Ham'
