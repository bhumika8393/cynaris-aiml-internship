import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest, f_regression

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("students.csv")

print("=" * 50)
print("Original Dataset")
print("=" * 50)
print(df)

# ==========================
# Label Encoding
# ==========================

print("\n===== Label Encoding =====")

label_encoder = LabelEncoder()

df["Gender_Label"] = label_encoder.fit_transform(df["Gender"])

print(df[["Gender", "Gender_Label"]])

# ==========================
# One Hot Encoding
# ==========================

print("\n===== One Hot Encoding =====")

one_hot = pd.get_dummies(df["Course"], prefix="Course")

print(one_hot)

# ==========================
# Ordinal Encoding
# ==========================

print("\n===== Ordinal Encoding =====")

attendance_category = []

for value in df["Attendance"]:
    if value >= 90:
        attendance_category.append("High")
    elif value >= 80:
        attendance_category.append("Medium")
    else:
        attendance_category.append("Low")

df["Attendance_Level"] = attendance_category

ordinal = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

df["Attendance_Ordinal"] = ordinal.fit_transform(
    df[["Attendance_Level"]]
)

print(df[["Attendance_Level", "Attendance_Ordinal"]])

# ==========================
# Scaling
# ==========================

X = df[["Attendance", "Marks"]]

standard = StandardScaler()
minmax = MinMaxScaler()
robust = RobustScaler()

standard_scaled = standard.fit_transform(X)
minmax_scaled = minmax.fit_transform(X)
robust_scaled = robust.fit_transform(X)

print("\n===== Standard Scaler =====")
print(standard_scaled)

print("\n===== MinMax Scaler =====")
print(minmax_scaled)

print("\n===== Robust Scaler =====")
print(robust_scaled)

# ==========================
# Feature Selection
# ==========================

print("\n===== SelectKBest =====")

features = pd.DataFrame()

features["Attendance"] = df["Attendance"]
features["Gender"] = df["Gender_Label"]
features = pd.concat([features, one_hot], axis=1)

target = df["Marks"]

selector = SelectKBest(score_func=f_regression, k="all")
selector.fit(features, target)

scores = pd.DataFrame({
    "Feature": features.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(
    by="Score",
    ascending=False
)

print(scores)

print("\nTop Features")

print(scores.head(5))

# ==========================
# Visualization
# ==========================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.hist(df["Marks"], bins=5)

plt.title("Before Scaling")

plt.subplot(1, 2, 2)

plt.hist(standard_scaled[:, 1], bins=5)

plt.title("After Standard Scaling")

plt.tight_layout()

plt.savefig("before_after_scaling.png")

plt.show()

print("\nGraph saved as before_after_scaling.png")

print("\nWeek 2 Day 1 Completed Successfully!")