import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("students_imbalanced.csv")

print("=" * 50)
print("Original Dataset")
print("=" * 50)
print(df)

# Encode categorical columns
le_gender = LabelEncoder()
le_course = LabelEncoder()
le_result = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Course"] = le_course.fit_transform(df["Course"])
df["Result"] = le_result.fit_transform(df["Result"])

X = df[["Gender", "Course", "Attendance", "Marks"]]
y = df["Result"]

print("\nOriginal Class Distribution")
print(y.value_counts())

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nBalanced Class Distribution")
print(pd.Series(y_resampled).value_counts())

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
y.value_counts().plot(kind="bar")
plt.title("Before SMOTE")

plt.subplot(1,2,2)
pd.Series(y_resampled).value_counts().plot(kind="bar")
plt.title("After SMOTE")

plt.tight_layout()
plt.savefig("smote_comparison.png")
plt.show()

print("\nSMOTE completed successfully!")