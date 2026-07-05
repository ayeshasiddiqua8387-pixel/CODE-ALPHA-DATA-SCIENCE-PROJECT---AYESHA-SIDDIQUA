import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\car data.csv")

# Convert categorical columns
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
df['Selling_type'] = df['Selling_type'].map({'Dealer':0,'Individual':1})
df['Transmission'] = df['Transmission'].map({'Manual':0,'Automatic':1})

df = df.drop('Car_Name', axis=1)

# Correlation matrix
corr = df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr)

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

print("Heatmap saved successfully!")