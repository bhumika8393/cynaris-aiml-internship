import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("students.csv")

print("="*50)
print("Original Dataset")
print("="*50)
print(df)

# ==========================
# Encode Categorical Columns
# ==========================

gender_encoder = LabelEncoder()
course_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Course"] = course_encoder.fit_transform(df["Course"])

# ==========================
# Features and Target
# ==========================

X = df[["Gender","Course","Attendance"]]
y = df["Marks"]

# ==========================
# Train Test Split
# ==========================

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :",len(X_train))
print("Testing Samples  :",len(X_test))

# ==========================
# Model Training
# ==========================

model = LinearRegression()

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)

print("\nTest Score (R²):",round(accuracy,3))

# ==========================
# Cross Validation
# ==========================

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")

print(scores)

print("\nAverage CV Score :",round(scores.mean(),3))

print("\nCompleted Successfully!")