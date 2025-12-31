import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data/dataset.csv")

df["input_text"] = df["skills"] + " " + df["education"] + " " + df["interests"]

X = df["input_text"]
y = df["career"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(X_train, y_train)

joblib.dump(model, "career_model.pkl")

print("Model Trained & Saved Successfully!")
