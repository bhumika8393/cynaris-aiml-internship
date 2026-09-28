import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation\n")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", auc)

# Save metrics
results = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

results.to_csv("evaluation_results.csv",index=False)

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test,y_pred)

plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(y_test,y_prob)

plt.savefig("roc_curve.png")
plt.close()

print("\nFiles saved successfully.")