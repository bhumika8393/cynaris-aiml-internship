

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Use a built-in dataset of movie reviews.
# Download is required the first time.
from sklearn.datasets import fetch_20newsgroups

# For a true sentiment dataset, use the NLTK movie_reviews corpus.
import nltk
from nltk.corpus import movie_reviews

try:
    movie_reviews.categories()
except LookupError:
    nltk.download("movie_reviews", quiet=True)

# Load positive and negative movie reviews.
texts = []
labels = []

for category, label in [("neg", 0), ("pos", 1)]:
    for file_id in movie_reviews.fileids(category):
        texts.append(movie_reviews.raw(file_id))
        labels.append(label)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

# Logistic Regression pipeline.
logistic_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
    ("model", LogisticRegression(max_iter=1000)),
])

# Random Forest pipeline.
random_forest_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )),
])

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}

os.makedirs("outputs", exist_ok=True)
results = {}

# Train, evaluate, and plot results for each model.
for name, model in models.items():
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    print(classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0,
    ))

    # Save confusion matrix.
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Negative", "Positive"],
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.close()

    # Save ROC curve.
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title(f"{name} - ROC Curve")
    plt.tight_layout()
    plt.savefig(
        f"outputs/{name.lower().replace(' ', '_')}_roc_curve.png"
    )
    plt.close()

# Compare model metrics.
print("\nMODEL COMPARISON")
print("-" * 70)
print(f"{'Model':<24}{'Accuracy':<14}{'Precision':<14}{'Recall':<14}")

for name, metrics in results.items():
    print(
        f"{name:<24}"
        f"{metrics['accuracy']:<14.3f}"
        f"{metrics['precision']:<14.3f}"
        f"{metrics['recall']:<14.3f}"
    )

print("\nPlots saved in the outputs/ folder.")