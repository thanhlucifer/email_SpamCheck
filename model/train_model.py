import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from preprocessing.text_cleaner import clean_text

def train_and_save_model(csv_path='data/spam.csv'):
    print("🔍 Đang đọc dữ liệu...")
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'text']
    except Exception as e:
        print(f"❌ Lỗi khi đọc dữ liệu: {e}")
        return

    print(f"📊 Số dòng dữ liệu ban đầu: {len(df)}")

    # Ánh xạ nhãn
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Loại bỏ NaN và các dòng không rõ nhãn
    df = df.dropna(subset=['label', 'text'])
    df = df[df['label'].isin([0, 1])]

    # Làm sạch văn bản
    print("🧹 Đang làm sạch dữ liệu...")
    df['cleaned'] = df['text'].apply(clean_text)
    df['cleaned'] = df['cleaned'].astype(str).str.strip()
    df = df[df['cleaned'] != '']
    df = df[~df['cleaned'].isna()]
    
    print(f"✅ Dữ liệu sau xử lý: {len(df)} dòng")

    # Nếu không còn dữ liệu hợp lệ thì dừng
    if df.empty:
        print("❌ Không còn dữ liệu hợp lệ sau khi xử lý.")
        return

    # Vector hoá
    print("🧠 Đang vector hoá dữ liệu...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']

    # Tách dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Huấn luyện model
    print("📈 Đang huấn luyện mô hình...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Đánh giá độ chính xác
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {acc * 100:.2f}%")

    # Lưu mô hình và vectorizer
    joblib.dump(model, 'model/spam_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    print("✅ Model saved to model/spam_model.pkl")
    print("✅ Vectorizer saved to model/vectorizer.pkl")

if __name__ == "__main__":
    train_and_save_model()
