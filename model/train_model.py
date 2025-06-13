import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from preprocessing.text_cleaner import clean_text

def train_and_save_model(csv_path='data/spam.csv'):
    df = pd.read_csv(csv_path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned'] = df['text'].apply(clean_text)

    X = CountVectorizer().fit_transform(df['cleaned'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f'Accuracy: {acc * 100:.2f}%')

    joblib.dump(model, 'model/spam_model.pkl')
    print("✅ Model saved to model/spam_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
