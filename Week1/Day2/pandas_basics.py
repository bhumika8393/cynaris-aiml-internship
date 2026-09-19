import pandas as pd

# Load dataset
df = pd.read_csv("indian_students.csv")

# Shape
print("Shape:")
print(df.shape)

# Data types
print("\nData Types:")
print(df.dtypes)

# First 10 rows
print("\nFirst 10 Rows:")
print(df.head(10))