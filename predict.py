import joblib

model = joblib.load("career_model.pkl")

skills = input("Enter your skills & interests: ")

prediction = model.predict([skills])[0]

print("\nSuggested Career:", prediction)
