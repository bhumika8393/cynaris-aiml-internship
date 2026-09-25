import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# ===============================
# Binary Classification
# ===============================

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("="*50)
print("Coefficients")
print("="*50)
print(model.coef_)

print("\nIntercept")
print(model.intercept_)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
auc = roc_auc_score(y_test,y_prob)

results = pd.DataFrame({
    "Accuracy":[accuracy],
    "Precision":[precision],
    "Recall":[recall],
    "F1 Score":[f1],
    "ROC AUC":[auc]
})

print("\nEvaluation Metrics")
print(results)

results.to_csv("breast_cancer_results.csv",index=False)

# ===============================
# Confusion Matrix
# ===============================

cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.savefig("confusion_matrix.png")

plt.show()

# ===============================
# ROC Curve
# ===============================

fpr,tpr,_ = roc_curve(y_test,y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,label=f"AUC={auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")

plt.show()

# ===============================
# Multi-Class Classification
# ===============================

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

ovr = LogisticRegression(
    multi_class="ovr",
    max_iter=5000
)

softmax = LogisticRegression(
    multi_class="multinomial",
    max_iter=5000
)

ovr.fit(X_train,y_train)
softmax.fit(X_train,y_train)

ovr_acc = accuracy_score(y_test,ovr.predict(X_test))
softmax_acc = accuracy_score(y_test,softmax.predict(X_test))

comparison = pd.DataFrame({

    "Model":[
        "One-vs-Rest",
        "Softmax"
    ],

    "Accuracy":[
        ovr_acc,
        softmax_acc
    ]

})

print("\nMulti-class Comparison")
print(comparison)

comparison.to_csv(
    "multiclass_results.csv",
    index=False
)

# ===============================
# Decision Boundary
# ===============================

X = iris.data[:, :2]
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    random_state=42
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train,y_train)

import numpy as np

x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1

xx,yy=np.meshgrid(
    np.arange(x_min,x_max,0.02),
    np.arange(y_min,y_max,0.02)
)

Z=model.predict(
    np.c_[xx.ravel(),yy.ravel()]
)

Z=Z.reshape(xx.shape)

plt.figure(figsize=(7,5))

plt.contourf(xx,yy,Z,alpha=0.3)

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    edgecolors="k"
)

plt.title("Decision Boundary")

plt.savefig("decision_boundary.png")

plt.show()

print("\nProject Completed Successfully!")