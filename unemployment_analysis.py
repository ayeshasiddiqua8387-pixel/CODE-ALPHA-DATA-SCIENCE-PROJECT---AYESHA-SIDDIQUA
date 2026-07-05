import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Unemployment in India.csv")
print("First 5 Rows of the Dataset:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nDataset Information:")
print(df.info())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Unemployment in India.csv")
df.columns = df.columns.str.strip()
print("Column Names after cleaning:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())
df = df.dropna()
print("\nDataset Shape After Removing Missing Values:")
print(df.shape)
print("\nDuplicate Rows:")
print(df.duplicated().sum())
print("\nStatistical Summary:")
print(df.describe())
plt.figure(figsize=(8, 5))
plt.hist(df["Estimated Unemployment Rate (%)"], bins=20)
plt.title("Distribution of Estimated Unemployment Rate")
plt.xlabel("Estimated Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
print("Histogram saved successfully!")
# State-wise Average Unemployment Rate

plt.figure(figsize=(14, 6))

sns.barplot(
    x="Region",
    y="Estimated Unemployment Rate (%)",
    data=df,
    estimator="mean"
)

plt.xticks(rotation=90)

plt.title("Average Unemployment Rate by Region")

plt.xlabel("Region")

plt.ylabel("Average Unemployment Rate (%)")

plt.tight_layout()
plt.savefig("bar_chart.png")
print("Bar chart saved successfully!")
# -----------------------------------
# Monthly Unemployment Trend
# -----------------------------------

# Remove spaces from dates
df["Date"] = df["Date"].str.strip()

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Sort data
df = df.sort_values("Date")

plt.figure(figsize=(12,6))

sns.lineplot(
    x="Date",
    y="Estimated Unemployment Rate (%)",
    data=df
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Estimated Unemployment Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("Monthly_Unemployment_Trend.png")

print("Monthly Trend graph saved successfully!")

plt.close()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("Correlation_Heatmap.png")
plt.close()
print("Heatmap saved")
# -----------------------------------
# KEY INSIGHTS
# -----------------------------------

print("\n===== KEY INSIGHTS =====")

print("\n1. Average Unemployment Rate:", df["Estimated Unemployment Rate (%)"].mean())

print("2. Highest Unemployment Rate:", df["Estimated Unemployment Rate (%)"].max())

print("3. Lowest Unemployment Rate:", df["Estimated Unemployment Rate (%)"].min())

print("\n4. Region with highest unemployment:")

print(df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False).head(1))

print("\n5. Region with lowest unemployment:")

print(df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values().head(1))