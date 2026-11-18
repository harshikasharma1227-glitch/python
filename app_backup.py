import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/fake_news_model.pkl")

# Page configuration
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide"
)

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Custom CSS
st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: white;
}

h1, h2, h3 {
    color: #60a5fa;
}

p {
    color: #cbd5e1;
}

.stTextArea textarea {
    background-color: #172033;
    color: white;
    border-radius: 12px;
}

.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
}

.stButton button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("🔎 TruthLens AI")

st.sidebar.write("AI-Powered News Analysis")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Home", "News Analyzer", "History", "About"]
)


# HOME PAGE
if page == "Home":

    st.title("🔎 TruthLens AI")

    st.subheader("Discover. Analyse. Verify.")

    st.write(
        "An AI-powered Fake News Detection System "
        "using Machine Learning and Natural Language Processing."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Articles", "44,898")

    with col2:
        st.metric("Test Accuracy", "98.65%")

    st.divider()

    st.subheader("What can TruthLens AI do?")

    st.write("🔍 Analyse news articles using Machine Learning.")

    st.write("🤖 Classify articles as Fake or Real.")

    st.write("📊 Display model confidence scores.")

    st.write("📜 Maintain a history of previous predictions.")


# NEWS ANALYZER
elif page == "News Analyzer":

    st.title("🔍 News Analyzer")

    st.write(
        "Paste a news article below to analyse it."
    )

    news = st.text_area(
        "Enter News Article",
        height=250,
        placeholder="Paste your news article here..."
    )

    if st.button("ANALYSE NEWS"):

        if not news.strip():

            st.warning("Please enter a news article.")

        else:

            prediction = model.predict([news])[0]

            probabilities = model.predict_proba([news])[0]

            class_index = list(model.classes_).index(prediction)

            confidence = probabilities[class_index] * 100

            if prediction == 0:

                st.error("⚠️ MODEL PREDICTION: FAKE NEWS")

                result = "FAKE NEWS"

            else:

                st.success("✅ MODEL PREDICTION: REAL NEWS")

                result = "REAL NEWS"

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence represents the model's estimated "
                "probability, not factual verification."
            )

            st.session_state.history.append({
                "News": news[:100],
                "Prediction": result,
                "Confidence": f"{confidence:.2f}%"
            })


# HISTORY PAGE
elif page == "History":

    st.title("📜 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info("No predictions available yet.")


# ABOUT PAGE
elif page == "About":

    st.title("ℹ️ About TruthLens AI")

    st.write("""
    TruthLens AI is a Fake News Detection project
    developed using Python and Machine Learning.

    The project uses Logistic Regression and
    TF-IDF Vectorization to classify news articles.

    TECHNOLOGIES USED:

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Natural Language Processing
    - Streamlit
    - Joblib

    The model provides predictions based on patterns
    learned from its training dataset. It does not
    independently verify news authenticity.
    """)

    st.info(
        "Developed as a Machine Learning project."
    )