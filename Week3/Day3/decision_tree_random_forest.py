import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ======================================
# Load Dataset
# ======================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names

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
# Decision Tree
# ======================================

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

tree.fit(X_train, y_train)

tree_prediction = tree.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_prediction
)

print("="*50)
print("Decision Tree Accuracy")
print("="*50)
print(tree_accuracy)

print("\nClassification Report\n")
print(classification_report(
    y_test,
    tree_prediction,
    target_names=iris.target_names
))

# ======================================
# Random Forest
# ======================================

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest.fit(X_train, y_train)

forest_prediction = forest.predict(X_test)

forest_accuracy = accuracy_score(
    y_test,
    forest_prediction
)

print("="*50)
print("Random Forest Accuracy")
print("="*50)
print(forest_accuracy)

print("\nClassification Report\n")
print(classification_report(
    y_test,
    forest_prediction,
    target_names=iris.target_names
))

# ======================================
# Cross Validation
# ======================================

tree_cv = cross_val_score(
    tree,
    X,
    y,
    cv=5
)

forest_cv = cross_val_score(
    forest,
    X,
    y,
    cv=5
)

results = pd.DataFrame({

    "Model":[
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy":[
        tree_accuracy,
        forest_accuracy
    ],

    "Cross Validation":[
        tree_cv.mean(),
        forest_cv.mean()
    ]

})

print(results)

results.to_csv(
    "iris_results.csv",
    index=False
)

# ======================================
# Feature Importance
# ======================================

importance = pd.Series(
    forest.feature_importances_,
    index=feature_names
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

# ======================================
# Decision Tree Visualization
# ======================================

plt.figure(figsize=(14,8))

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.savefig("decision_tree.png")

plt.show()

print("\nProject Completed Successfully!")