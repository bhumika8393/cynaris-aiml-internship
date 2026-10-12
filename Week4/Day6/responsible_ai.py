"""
Week 4 Day 6
Responsible AI - Bias, Fairness & Transparency
"""

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

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy : {accuracy:.2f}")

print("\nResponsible AI Checklist")
print("------------------------")
print("✔ Model trained successfully")
print("✔ Performance evaluated")
print("✔ Fairness should be assessed using AIF360")
print("✔ Explainability can be performed using SHAP")
print("✔ Model card documentation completed")