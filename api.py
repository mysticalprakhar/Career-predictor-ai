from fastapi import FastAPI
import joblib

model = joblib.load("career_model.pkl")
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Career Predictor API is Live"}

@app.get("/predict")
def predict(skills: str):
    career = model.predict([skills])[0]
    return {"career": career}
