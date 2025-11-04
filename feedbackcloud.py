import re
import pandas as pd
import numpy as np
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK sentiment model once
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="Live Feedback NLP Dashboard", page_icon="🟡", layout="wide")
st.markdown("<h1 style='text-align: center; font-size: 80px;'>Live Feedback Insight Board</h1>", unsafe_allow_html=True)

# Get published CSV URL
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTj2RxTSpCcbWntncEHgmMcl13hfgc56_f_hw2OnRXY8JlV74zgzOU27Gb6gea04XJWdnjP-inhIfnd/pub?gid=1351743539&single=true&output=csv"

# Load CSV data
@st.cache_data(ttl=60)
def load_data(url):
    return pd.read_csv(url)

try:
    df = load_data(CSV_URL)
except Exception as e:
    st.error("Couldn't load Google Sheet CSV. Check your URL in secrets.toml.")
    st.stop()

# Normalize headers
df.columns = [c.strip() for c in df.columns]

# Your column names from Google Form
TIMESTAMP_COL = "timestamp"
REVIEW_COL = "comment"

# Validate columns
missing = [c for c in [TIMESTAMP_COL, REVIEW_COL] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}\n\nFound: {list(df.columns)}")
    st.stop()

# Rename to standardized internal names
df = df.rename(columns={
    TIMESTAMP_COL: "timestamp",
    REVIEW_COL: "review_text"
})

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["category"] = ""  # placeholder for future categories

# Clean text
def clean_text(txt):
    txt = str(txt).lower()
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

df["clean"] = df["review_text"].apply(clean_text)

# Sentiment
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["review_text"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Wordcloud
def generate_wordcloud(texts):
    text = " ".join(texts)
    wc = WordCloud(width=1000, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# Clustering
def cluster_texts(texts, n_clusters=3):
    if len(texts) < n_clusters:
        return np.zeros(len(texts)), ["General Feedback"]

    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    top_terms = []
    for i in range(n_clusters):
        idx = np.argsort(kmeans.cluster_centers_[i])[::-1][:7]
        top_terms.append(", ".join(terms[idx]))

    return labels, top_terms

labels, topics = cluster_texts(df["clean"].tolist())
df["segment"] = labels

# Layout
col1, col2 = st.columns([3,2])

with col1:
    st.subheader("Word Cloud")
    fig = generate_wordcloud(df["clean"])
    st.pyplot(fig)

with col2:
    st.subheader("Sentiment Overview")
    avg_sent = df["sentiment"].mean()
    st.metric("Avg Sentiment Score", f"{avg_sent:.2f}")
    st.write("Range: -1 negative → +1 positive")
    st.progress(int(((avg_sent + 1) / 2) * 100))

st.subheader("Feedback Segments")
for i, topic in enumerate(topics):
    st.write(f"**Segment {i}:** {topic}")
    st.dataframe(df[df["segment"] == i][["timestamp", "review_text", "sentiment"]])

st.subheader("Latest Responses")
st.dataframe(df.sort_values("timestamp", ascending=False)[["timestamp", "review_text", "sentiment"]].head(30))
