import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "fake_news_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f6fa;
    color: #17233c;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    background: #10233f;
    padding: 45px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    color: white;
}

.hero p {
    font-size: 17px;
    color: #dce5f2;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #10233f;
    margin-top: 25px;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e0e5ed;
    margin-bottom: 15px;
}

.result-real {
    background: #e4f7ed;
    border-left: 6px solid #159957;
    padding: 25px;
    border-radius: 12px;
}

.result-fake {
    background: #fdeaea;
    border-left: 6px solid #dc3545;
    padding: 25px;
    border-radius: 12px;
}

div.stButton > button {
    background-color: #164e91;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #0d3566;
    color: white;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🔎 TruthLens AI")
    st.caption("AI-Powered News Analysis")
    st.divider()

    pages = ["Home", "News Analyzer", "History", "About"]

    for page in pages:
        if st.button(page, use_container_width=True):
            st.session_state.page = page

    st.divider()
    st.caption("A student AI/ML project")

# ---------------- HOME PAGE ----------------
if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>See Beyond the Headlines.</h1>
        <p>
        Welcome to TruthLens AI — an educational project
        that uses Machine Learning to analyse news articles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Explore TruthLens</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>📰 News Analysis</h3>
        <p>Submit a news article and let the trained model analyse its text.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>🤖 Machine Learning</h3>
        <p>Uses TF-IDF and Logistic Regression for text classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
        <h3>📊 Analysis History</h3>
        <p>Review the articles analysed during your current session.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("Go to News Analyzer from the sidebar to test an article.")

    st.warning(
        "Important: This model predicts patterns learned from its training "
        "dataset. It does not independently verify whether a news story is true."
    )

# ---------------- NEWS ANALYZER ----------------
elif st.session_state.page == "News Analyzer":

    st.title("📰 News Analyzer")
    st.write("Paste a news article below to analyse it.")

    news_text = st.text_area(
        "Enter News Article",
        height=250,
        placeholder="Paste your news article here..."
    )

    if st.button("🔍 Analyse News"):

        if len(news_text.strip()) < 50:
            st.warning("Please enter a longer article (at least 50 characters).")

        else:
            prediction = model.predict([news_text])[0]
            probabilities = model.predict_proba([news_text])[0]
            confidence = max(probabilities) * 100

            if prediction == 1:
                st.markdown(f"""
                <div class="result-real">
                <h2>🟢 Model Prediction: REAL</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                label = "REAL"

            else:
                st.markdown(f"""
                <div class="result-fake">
                <h2>🔴 Model Prediction: FAKE</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                label = "FAKE"

            st.session_state.history.append({
                "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Prediction": label,
                "Confidence": f"{confidence:.2f}%"
            })

            st.caption(
                "This is a machine-learning prediction, not a verified fact-check."
            )

# ---------------- HISTORY PAGE ----------------
elif st.session_state.page == "History":

    st.title("📊 Analysis History")
    st.write("Articles analysed during this session.")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No articles analysed yet.")

# ---------------- ABOUT PAGE ----------------
elif st.session_state.page == "About":

    st.title("About TruthLens AI")

    st.markdown("""
    <div class="info-card">
    <h2>🔎 TruthLens AI</h2>
    <p>
    TruthLens AI is a Fake News Detection project developed
    using Python and Machine Learning.
    </p>
    <h3>Technologies Used</h3>
    <ul>
        <li>Python</li>
        <li>Streamlit</li>
        <li>Pandas</li>
        <li>Scikit-learn</li>
        <li>TF-IDF Vectorization</li>
        <li>Logistic Regression</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "The project is designed for educational purposes. "
        "Its predictions should not be treated as proof of authenticity."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption("© 2026 TruthLens AI | Learn. Analyse. Think Critically.")
