import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# ==========================================
# Load Dataset
# ==========================================

data = load_wine()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("="*50)
print("Wine Dataset")
print("="*50)
print(X.head())

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# SVM GridSearch
# ==========================================

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])

svm_params = {
    "model__C":[0.1,1,10,100],
    "model__kernel":["linear","rbf"],
    "model__gamma":["scale","auto"]
}

grid = GridSearchCV(
    svm_pipeline,
    svm_params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train,y_train)

svm_pred = grid.predict(X_test)

svm_accuracy = accuracy_score(y_test,svm_pred)

print("\nBest SVM Parameters")
print(grid.best_params_)
print("Accuracy:",svm_accuracy)

# ==========================================
# KNN Random Search
# ==========================================

knn_pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("model",KNeighborsClassifier())
])

knn_params = {
    "model__n_neighbors":[3,5,7,9,11],
    "model__weights":["uniform","distance"],
    "model__metric":["euclidean","manhattan"]
}

random = RandomizedSearchCV(
    knn_pipeline,
    knn_params,
    n_iter=8,
    cv=5,
    random_state=42,
    scoring="accuracy"
)

random.fit(X_train,y_train)

knn_pred = random.predict(X_test)

knn_accuracy = accuracy_score(y_test,knn_pred)

print("\nBest KNN Parameters")
print(random.best_params_)
print("Accuracy:",knn_accuracy)

# ==========================================
# Comparison Table
# ==========================================

results = pd.DataFrame({
    "Model":[
        "SVM GridSearch",
        "KNN RandomSearch"
    ],
    "Accuracy":[
        svm_accuracy,
        knn_accuracy
    ]
})

print("\nModel Comparison")
print(results)

results.to_csv("model_comparison.csv",index=False)

print("\nSaved model_comparison.csv")