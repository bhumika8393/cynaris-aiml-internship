import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("student_visualization.csv")

# Set style
sns.set_style("whitegrid")

# -----------------------
# Histogram
# -----------------------
plt.figure(figsize=(6,4))
sns.histplot(df["Marks"], bins=5, kde=True)
plt.title("Marks Distribution")
plt.savefig("plots/marks_distribution.png")
plt.close()

# -----------------------
# Box Plot
# -----------------------
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Marks"])
plt.title("Marks Box Plot")
plt.savefig("plots/marks_boxplot.png")
plt.close()

# -----------------------
# Scatter Plot
# -----------------------
plt.figure(figsize=(6,4))
sns.scatterplot(data=df,x="Attendance",y="Marks",hue="Course")
plt.title("Attendance vs Marks")
plt.savefig("plots/attendance_vs_marks.png")
plt.close()

# -----------------------
# Correlation Heatmap
# -----------------------
plt.figure(figsize=(6,4))
corr=df[["Marks","Attendance","Age"]].corr()

sns.heatmap(corr,annot=True,cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("plots/correlation_heatmap.png")

plt.close()

# -----------------------
# Count Plot
# -----------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df,x="Course")

plt.title("Students Per Course")

plt.savefig("plots/course_count.png")

plt.close()

print("All graphs created successfully.")