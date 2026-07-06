# ==========================================
# IRIS FLOWER CLASSIFICATION USING KAGGLE CSV
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# Step 1: Load Dataset
# ==========================================

df = pd.read_csv(r"C:\Users\acer\Downloads\Iris.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

# ==========================================
# Step 2: Remove Id Column
# ==========================================

if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# ==========================================
# Step 3: Features and Target
# ==========================================

X = df.drop("Species", axis=1)
y = df["Species"]

# ==========================================
# Step 4: Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# ==========================================
# Step 5: Create Model
# ==========================================

model = DecisionTreeClassifier(random_state=42)

# ==========================================
# Step 6: Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================
# Step 7: Predict
# ==========================================

y_pred = model.predict(X_test)

print("\n========== PREDICTIONS ==========")
print(y_pred)

# ==========================================
# Step 8: Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)

# ==========================================
# Step 9: Classification Report
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 10: Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

# Save the confusion matrix
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

print("Confusion Matrix saved successfully!")

# Show the figure
plt.show()
# ==========================================
# Step 11: Predict New Flower
# ==========================================

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\n========== NEW FLOWER PREDICTION ==========")
print("Predicted Species:", prediction[0])

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")