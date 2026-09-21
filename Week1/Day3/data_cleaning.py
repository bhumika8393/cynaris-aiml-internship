import pandas as pd

# Load dataset
df = pd.read_csv("employees.csv")

print("Original Dataset")
print(df)

# Dataset information
print("\nDataset Info")
print(df.info())

# Missing values
print("\nMissing Values")
print(df.isnull().sum())

# Fill missing salary with average salary
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# Fill missing age with average age
df["Age"] = df["Age"].fillna(df["Age"].mean())

print("\nCleaned Dataset")
print(df)

# Save cleaned data
df.to_csv("cleaned_employees.csv", index=False)

print("\nCleaning completed successfully.")