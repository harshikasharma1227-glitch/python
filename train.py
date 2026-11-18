import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# STEP 1: Load datasets

print("Loading datasets...")

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")


# STEP 2: Assign labels

fake["label"] = 0
real["label"] = 1


# STEP 3: Combine datasets

data = pd.concat([fake, real], ignore_index=True)

print("Total articles:", len(data))


# STEP 4: Prepare text

data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)

data = data[data["content"].str.strip() != ""]

data = data.drop_duplicates(subset="content")


X = data["content"]
y = data["label"]


# STEP 5: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# STEP 6: Create ML pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])


# STEP 7: Train model

print("Training model...")

model.fit(X_train, y_train)

print("Training completed!")


# STEP 8: Test model

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# STEP 9: Save model

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")

print("\nModel saved successfully!")