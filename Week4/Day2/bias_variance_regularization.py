import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Different regularization strengths
C_values = [0.01, 0.1, 1, 10, 100]

results = []

print("\nRegularization Comparison\n")

for C in C_values:

    model = LogisticRegression(
        C=C,
        penalty="l2",
        solver="liblinear",
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append([C, accuracy])

    print(f"C={C:<6} Accuracy={accuracy:.4f}")

# Save results
df = pd.DataFrame(
    results,
    columns=["C Value", "Accuracy"]
)

df.to_csv("model_comparison.csv", index=False)

# Plot
plt.figure(figsize=(8,5))

plt.plot(
    df["C Value"],
    df["Accuracy"],
    marker="o"
)

plt.xscale("log")

plt.xlabel("Regularization Strength (C)")
plt.ylabel("Accuracy")

plt.title("Bias-Variance Tradeoff using L2 Regularization")

plt.grid(True)

plt.savefig("regularization_accuracy.png")

plt.show()

print("\nFiles saved successfully.")