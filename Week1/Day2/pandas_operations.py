import pandas as pd
import os

# Load the dataset
df = pd.read_csv("indian_students.csv")

# -------------------------------
# Filter Operation
# -------------------------------
# Display students with marks greater than 85
filtered = df[df["Marks"] > 85]

print("Students with Marks > 85")
print(filtered)

# -------------------------------
# GroupBy Operation
# -------------------------------
# Calculate average marks for each course
grouped = df.groupby("Course")["Marks"].mean()

print("\nAverage Marks by Course")
print(grouped)

# -------------------------------
# Merge Operation
# -------------------------------
# Create another DataFrame
course_info = pd.DataFrame({
    "Course": ["AIML", "CSE", "ISE"],
    "Duration": ["4 Years", "4 Years", "4 Years"]
})

# Merge both DataFrames
merged = pd.merge(df, course_info, on="Course")

print("\nMerged Data")
print(merged)

# -------------------------------
# Pivot Table Operation
# -------------------------------
pivot = pd.pivot_table(
    df,
    values="Marks",
    index="State",
    columns="Course",
    aggfunc="mean"
)

print("\nPivot Table")
print(pivot)

# -------------------------------
# Export Files
# -------------------------------
filtered.to_csv("cleaned_students.csv", index=False)
filtered.to_parquet("cleaned_students.parquet", index=False)

# Compare file sizes
csv_size = os.path.getsize("cleaned_students.csv")
parquet_size = os.path.getsize("cleaned_students.parquet")

print("\nCSV File Size:", csv_size, "bytes")
print("Parquet File Size:", parquet_size, "bytes")