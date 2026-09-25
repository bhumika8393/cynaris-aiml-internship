import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ======================================
# Load Dataset
# ======================================

iris = load_iris()

X = iris.data
y = iris.target

# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# Feature Scaling
# ======================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================
# SVM Model
# ======================================

svm = SVC(kernel="rbf", random_state=42)

svm.fit(X_train, y_train)

svm_prediction = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_prediction)

print("=" * 50)
print("SVM Accuracy")
print("=" * 50)
print(svm_accuracy)

print(classification_report(
    y_test,
    svm_prediction,
    target_names=iris.target_names
))

# ======================================
# KNN Model
# ======================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_prediction = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_prediction)

print("=" * 50)
print("KNN Accuracy")
print("=" * 50)
print(knn_accuracy)

print(classification_report(
    y_test,
    knn_prediction,
    target_names=iris.target_names
))

# ======================================
# Save Results
# ======================================

results = pd.DataFrame({

    "Model":[
        "SVM",
        "KNN"
    ],

    "Accuracy":[
        svm_accuracy,
        knn_accuracy
    ]

})

print(results)

results.to_csv(
    "model_comparison.csv",
    index=False
)

# ======================================
# Confusion Matrix
# ======================================

cm = confusion_matrix(
    y_test,
    svm_prediction
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("SVM Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# ======================================
# Accuracy Comparison
# ======================================

plt.figure(figsize=(6,4))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.savefig("accuracy_comparison.png")

plt.show()

print("\nProject Completed Successfully!")
