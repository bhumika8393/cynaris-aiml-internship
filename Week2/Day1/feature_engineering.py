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

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv("students.csv")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df)

# ==================================================
# Label Encoding
# ==================================================

print("\n===== LABEL ENCODING =====")

label = LabelEncoder()

df["Gender_Label"] = label.fit_transform(df["Gender"])

print(df[["Gender", "Gender_Label"]])

# ==================================================
# One Hot Encoding
# ==================================================

print("\n===== ONE HOT ENCODING =====")

course_encoded = pd.get_dummies(df["Course"], prefix="Course")

print(course_encoded)

# ==================================================
# Ordinal Encoding
# ==================================================

print("\n===== ORDINAL ENCODING =====")

attendance_level = []

for value in df["Attendance"]:
    if value >= 90:
        attendance_level.append("High")
    elif value >= 80:
        attendance_level.append("Medium")
    else:
        attendance_level.append("Low")

df["Attendance_Level"] = attendance_level

ordinal = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

df["Attendance_Ordinal"] = ordinal.fit_transform(
    df[["Attendance_Level"]]
)

print(df[["Attendance_Level", "Attendance_Ordinal"]])

# ==================================================
# Scaling
# ==================================================

print("\n===== SCALING =====")

standard = StandardScaler()
minmax = MinMaxScaler()
robust = RobustScaler()

# Scale only Marks column
df["Marks_StandardScaler"] = standard.fit_transform(df[["Marks"]])

df["Marks_MinMaxScaler"] = minmax.fit_transform(df[["Marks"]])

df["Marks_RobustScaler"] = robust.fit_transform(df[["Marks"]])

print(df[[
    "Marks",
    "Marks_StandardScaler",
    "Marks_MinMaxScaler",
    "Marks_RobustScaler"
]])

# ==================================================
# Feature Selection
# ==================================================

print("\n===== SELECT KBEST =====")

features = pd.DataFrame()

features["Attendance"] = df["Attendance"]
features["Gender"] = df["Gender_Label"]

features = pd.concat(
    [features, course_encoded],
    axis=1
)

target = df["Marks"]

selector = SelectKBest(
    score_func=f_regression,
    k="all"
)

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

print("\nTop 5 Features")
print(scores.head())

# ==================================================
# Visualization
# ==================================================

plt.figure(figsize=(14,8))

# Original Marks
plt.subplot(2,2,1)
plt.hist(df["Marks"], bins=5, edgecolor="black")
plt.title("Original Marks")
plt.xlabel("Marks")
plt.ylabel("Students")

# Standard Scaler
plt.subplot(2,2,2)
plt.hist(df["Marks_StandardScaler"], bins=5, edgecolor="black")
plt.title("StandardScaler")
plt.xlabel("Scaled Marks")
plt.ylabel("Students")

# MinMax Scaler
plt.subplot(2,2,3)
plt.hist(df["Marks_MinMaxScaler"], bins=5, edgecolor="black")
plt.title("MinMaxScaler")
plt.xlabel("Scaled Marks")
plt.ylabel("Students")

# Robust Scaler
plt.subplot(2,2,4)
plt.hist(df["Marks_RobustScaler"], bins=5, edgecolor="black")
plt.title("RobustScaler")
plt.xlabel("Scaled Marks")
plt.ylabel("Students")

plt.tight_layout()

plt.savefig("before_after_scaling.png")

plt.show()

print("\nGraph saved as before_after_scaling.png")