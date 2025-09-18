# 1. IMPORT LIBRARIES
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

# --- Optional Word2Vec ---
try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except Exception:
    print("⚠ gensim not supported in Python 3.13, skipping Word2Vec.")
    HAS_GENSIM = False

# 2. DOWNLOAD NLTK RESOURCES (safe check)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠ spaCy model not found. Downloading...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# 3. LOAD DATASET
df = pd.read_excel("data/reviews.xlsx", engine="openpyxl")

text_col = "Review Text"
rating_col = "Rating"
df = df[[text_col, rating_col]].dropna()
df = df.rename(columns={text_col: "text", rating_col: "rating"})

# If a date column exists, keep it for trend analysis
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 4. DATA PREPROCESSING
def preprocess(text, do_stemming=False):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1
    ]
    if do_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)

df["clean"] = df["text"].apply(preprocess)

# 5. SENTIMENT MAPPING
def map_sentiment(r):
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(map_sentiment)

# 6. WORD CLOUDS
for s in ["Positive", "Neutral", "Negative"]:
    all_text = " ".join(df[df["sentiment"] == s]["clean"])
    if all_text.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(
            all_text
        )
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{s} Reviews Word Cloud")
        plt.show()

# 7. N-GRAM ANALYSIS
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words="english")
X_ngrams = vectorizer.fit_transform(df["clean"])
ngrams_freq = X_ngrams.sum(axis=0)
ngrams = [(ng, ngrams_freq[0, idx]) for ng, idx in vectorizer.vocabulary_.items()]
ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)[:20]
print("Top 20 N-grams:", ngrams)

# 8. SENTIMENT CLASSIFICATION
tfidf = TfidfVectorizer(max_features=5000)
X_vect = tfidf.fit_transform(df["clean"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42
)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
}

for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Neutral", "Negative"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positive", "Neutral", "Negative"],
                yticklabels=["Positive", "Neutral", "Negative"])
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # ROC / AUC (only works for binary classification)
    if len(set(y)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label="Positive")
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

# 9. TOPIC MODELING (LDA)
lda_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
lda_matrix = lda_vectorizer.fit_transform(df["clean"])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(lda_matrix)

terms = lda_vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"\nTopic {idx}:")
    print([terms[i] for i in topic.argsort()[-10:]])

# 10. TREND ANALYSIS (if Date available)
if "Date" in df.columns:
    sentiment_over_time = (
        df.groupby([pd.Grouper(key="Date", freq="M"), "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.line(sentiment_over_time, x="Date", y="count", color="sentiment",
                  title="Sentiment Over Time")
    fig.show()

# 11. NAMED ENTITY RECOGNITION (NER)
sample_texts = df["text"].sample(min(5, len(df)), random_state=42).tolist()
for txt in sample_texts:
    doc = nlp(txt)
    print(f"\nReview: {txt}")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

# 12. ACTIONABLE INSIGHTS
print("\n=== Actionable Insights ===")
pos_words = " ".join(df[df["sentiment"] == "Positive"]["clean"]).split()
neg_words = " ".join(df[df["sentiment"] == "Negative"]["clean"]).split()

top_pos = Counter(pos_words).most_common(10)
top_neg = Counter(neg_words).most_common(10)

print(" Most praised aspects:", top_pos)
print(" Most criticized aspects:", top_neg)
