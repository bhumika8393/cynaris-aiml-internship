import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("titanic_sample.csv")

print("=" * 50)
print("Original Dataset")
print("=" * 50)
print(df)

# Features and Target

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Numeric and Categorical Columns

numeric_features = ["Age", "Fare", "Pclass"]

categorical_features = ["Sex"]

# Numeric Pipeline

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine Pipelines

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Apply Transformation

X_processed = preprocessor.fit_transform(X)

# Column Names

encoded_columns = preprocessor.named_transformers_["cat"] \
    .named_steps["encoder"] \
    .get_feature_names_out(categorical_features)

column_names = numeric_features + list(encoded_columns)

processed_df = pd.DataFrame(
    X_processed,
    columns=column_names
)

processed_df["Survived"] = y.values

print("\nProcessed Dataset")
print(processed_df)

# Save Dataset

processed_df.to_csv(
    "processed_titanic.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")