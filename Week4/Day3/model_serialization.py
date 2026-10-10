"""
Week 4 Day 3
Model Serialization using Pickle and Joblib
"""

import pickle
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=300)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy : {accuracy:.2f}")

# Save using Pickle
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved using Pickle.")

# Save using Joblib
joblib.dump(model, "logistic_model.joblib")

print("Model saved using Joblib.")

# Load Pickle model
with open("logistic_model.pkl", "rb") as file:
    pickle_model = pickle.load(file)

# Load Joblib model
joblib_model = joblib.load("logistic_model.joblib")

print("Pickle model prediction:")
print(pickle_model.predict([X_test[0]]))

print("Joblib model prediction:")
print(joblib_model.predict([X_test[0]]))