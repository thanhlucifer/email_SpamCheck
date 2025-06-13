# 📧 Email Spam Check with Streamlit

A simple web application that uses **Naive Bayes Classification** to detect whether an email is **Spam** or **Ham**. Built with `Python`, `Scikit-learn`, `NLTK`, and `Streamlit`.

---

## 🔍 Features

- Clean and preprocess text using `nltk`.
- Train a spam classification model using `MultinomialNB`.
- Save and load model with `joblib`.
- Web interface with `Streamlit` for live spam checking.

---

## 🧠 Model

- **Algorithm**: Multinomial Naive Bayes
- **Text Processing**: CountVectorizer
- **Training Data**: `spam.csv` (from UCI dataset)

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/thanhlucifer/email_SpamCheck.git
cd email_SpamCheck
```

### 2. Create virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

You can train the model manually using:

```bash
python main.py
```

This will:
- Load and clean `data/spam.csv`
- Train a Naive Bayes model
- Save the model to `model/spam_model.pkl`
- Save the vectorizer to `model/vectorizer.pkl`

---

## 🌐 Run the Web App

```bash
streamlit run app/app.py
```

Then open the provided `localhost` link in your browser.

---

## 🧼 Preprocessing

Located in: `preprocessing/text_cleaner.py`

Main functions:
- Lowercasing
- Removing punctuation, stopwords
- Tokenization using `nltk`

---

## 🖼 Folder Structure

```
email_SpamCheck/
├── app/
│   └── app.py              # Streamlit app
├── data/
│   └── spam.csv            # Training dataset
├── model/
│   ├── spam_model.pkl      # Trained model
│   └── vectorizer.pkl      # Fitted CountVectorizer
├── preprocessing/
│   ├── __init__.py
│   └── text_cleaner.py     # Text cleaning utilities
├── main.py                 # Train model entry point
├── requirements.txt
└── README.md
```

---

## 🧪 Sample Test

Try inputting:

```
Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!
```

→ Output: **Spam**

```
Hi, just checking in to see if you're free for lunch today.
```

→ Output: **Ham**

---

## 👨‍💻 Author

**Thành Phạm**  
📍 Bình Định, Việt Nam  
💼 [GitHub](https://github.com/thanhlucifer)

---

## 📄 License

This project is open source and available under the MIT License.