import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import streamlit as st
import plotly.graph_objects as go

# ---------- Normalize User Input ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    try:
        train = pd.read_csv("dataset/training.csv")
        test = pd.read_csv("dataset/test.csv")
        val = pd.read_csv("dataset/validation.csv")
        return train, test, val
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None, None, None

# ---------- Train Model and Vectorizer ----------
@st.cache_resource
def train_model(train_df, test_df):
    emotion_labels = {
        0: '😢 Sadness', 1: '😊 Joy', 2: '❤️ Love',
        3: '😠 Anger', 4: '😨 Fear', 5: '😲 Surprise'
    }
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf.fit_transform(train_df['text'].fillna(''))
    X_test = tfidf.transform(test_df['text'].fillna(''))
    y_train, y_test = train_df['label'], test_df['label']

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, tfidf, cm, acc, y_test, y_pred, emotion_labels

# ---------- Visualize Confusion Matrix ----------
def plot_confusion_matrix(cm, emotion_labels):
    labels = [label.split(' ', 1)[1] for label in emotion_labels.values()]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale='Blues', text=cm, texttemplate="%{text}"
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

# ---------- Show Confidence ----------
def show_confidence(emotion_labels, probs):
    for label, p in zip(emotion_labels.values(), probs):
        st.write(f"{label}: {p:.2%}")
        st.progress(float(p))

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Emotion AI Classifier", layout="centered")
    st.title("🎭 Emotion AI Classifier")
    st.markdown("Classifies text into 6 emotions using TF-IDF + Logistic Regression.")

    with st.expander("🧠 Methodology"):
        st.markdown("""
        - **TF-IDF** vectorization with top 5000 features
        - **Logistic Regression** trained on labeled emotion dataset
        - Live predictions with confidence scores
        """)

    train_df, test_df, val_df = load_data()
    if train_df is None: st.stop()

    with st.spinner("🔄 Training model..."):
        model, vectorizer, cm, acc, y_test, y_pred, labels = train_model(train_df, test_df)
    st.success("✅ Model trained successfully!")

    # ---------- Metrics ----------
    st.metric("Accuracy", f"{acc:.2%}")
    st.plotly_chart(plot_confusion_matrix(cm, labels))

    st.subheader("🔍 Try Custom Text")
    user_input = st.text_area("Enter a sentence")
    if st.button("Analyze"):
        if user_input:
            vec = vectorizer.transform([preprocess(user_input)])
            pred = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]
            st.write("**Detected Emotion:**", labels[pred])
            show_confidence(labels, probs)
        else:
            st.warning("⚠️ Please enter some text.")

    # ---------- Example Test ----------
    st.subheader("🧪 Example Sentences")
    examples = {
        '😢 Sadness': "I miss my best friend so much.",
        '😊 Joy': "I just got my dream job!",
        '❤️ Love': "Being with you is the best feeling ever.",
        '😠 Anger': "I'm furious at what they did!",
        '😨 Fear': "I’m scared to walk home alone.",
        '😲 Surprise': "I didn't expect that at all!"
    }
    selected = st.selectbox("Choose an emotion example:", list(examples.keys()))
    if st.button("Run Example"):
        sentence = examples[selected]
        vec = vectorizer.transform([preprocess(sentence)])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        st.write("**Detected Emotion:**", labels[pred])
        st.write(f"**Input:** _{sentence}_")
        show_confidence(labels, probs)

if __name__ == "__main__":
    main()
