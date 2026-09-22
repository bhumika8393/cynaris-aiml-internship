import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("students.csv")

print("=" * 60)
print("Original Dataset")
print("=" * 60)
print(df)

# ==========================================
# Encode Gender
# ==========================================

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# ==========================================
# Features and Target
# ==========================================

X = df[["Attendance", "Gender"]]
y = df["Marks"]

# ==========================================
# Standard Scaling
# ==========================================

standard = StandardScaler()

standard_scaled = standard.fit_transform(X)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=["Attendance", "Gender"]
)

print("\nStandard Scaled Data")
print(standard_df)

# ==========================================
# MinMax Scaling
# ==========================================

minmax = MinMaxScaler()

minmax_scaled = minmax.fit_transform(X)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=["Attendance", "Gender"]
)

print("\nMinMax Scaled Data")
print(minmax_df)

# ==========================================
# Robust Scaling
# ==========================================

robust = RobustScaler()

robust_scaled = robust.fit_transform(X)

robust_df = pd.DataFrame(
    robust_scaled,
    columns=["Attendance", "Gender"]
)

print("\nRobust Scaled Data")
print(robust_df)

# ==========================================
# Feature Selection
# ==========================================

selector = SelectKBest(score_func=f_regression, k="all")

selector.fit(X, y)

scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(
    by="Score",
    ascending=False
)

print("\n")
print("=" * 60)
print("Feature Scores")
print("=" * 60)

print(scores)

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(14,10))

# Original Attendance

plt.subplot(2,2,1)
plt.hist(
    df["Attendance"],
    bins=6,
    edgecolor="black"
)
plt.title("Original Attendance")

# StandardScaler

plt.subplot(2,2,2)
plt.hist(
    standard_df["Attendance"],
    bins=6,
    edgecolor="black"
)
plt.title("StandardScaler")

# MinMaxScaler

plt.subplot(2,2,3)
plt.hist(
    minmax_df["Attendance"],
    bins=6,
    edgecolor="black"
)
plt.title("MinMaxScaler")

# RobustScaler

plt.subplot(2,2,4)
plt.hist(
    robust_df["Attendance"],
    bins=6,
    edgecolor="black"
)
plt.title("RobustScaler")

plt.tight_layout()

plt.savefig("scaling_comparison.png")

plt.show()

print("\nScaling comparison image saved as scaling_comparison.png")