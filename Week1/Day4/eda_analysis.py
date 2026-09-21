import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students_eda.csv")

print("========== DATA INFO ==========")
print(df.info())

print("\n========== DESCRIPTION ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ------------------------
# Observation 1
# ------------------------
print("\nAverage Marks:", df["Marks"].mean())

# Observation 2
print("Average Attendance:", df["Attendance"].mean())

# Observation 3
print("Highest Marks:", df["Marks"].max())

# Observation 4
print("Lowest Marks:", df["Marks"].min())

# Observation 5
print("Courses:")
print(df["Course"].value_counts())

# ------------------------
# Distribution Plots
# ------------------------

plt.figure(figsize=(6,4))
df["Marks"].hist()
plt.title("Marks Distribution")
plt.savefig("plots/marks_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
df["Age"].hist()
plt.title("Age Distribution")
plt.savefig("plots/age_distribution.png")
plt.close()

# ------------------------
# Correlation Heatmap
# ------------------------

corr = df[["Age","Marks","Attendance"]].corr()

plt.figure(figsize=(5,4))
plt.imshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# ------------------------
# Category Count
# ------------------------

plt.figure(figsize=(6,4))
df["Course"].value_counts().plot(kind="bar")
plt.title("Course Count")
plt.savefig("plots/course_count.png")
plt.close()

# Save cleaned dataset
df.to_csv("cleaned_students.csv", index=False)

print("\nEDA Completed Successfully!")