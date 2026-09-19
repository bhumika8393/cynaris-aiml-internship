import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("students.csv")

print("Dataset:")
print(df)

# Convert only numeric columns to a NumPy array
marks = df[["Math", "Science", "English"]].to_numpy()

# Mean
print("\nMean:")
print(np.mean(marks, axis=0))

# Standard Deviation
print("\nStandard Deviation:")
print(np.std(marks, axis=0))

# Correlation Matrix
print("\nCorrelation Matrix:")
print(np.corrcoef(marks, rowvar=False))