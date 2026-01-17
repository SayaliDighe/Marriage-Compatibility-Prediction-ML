# Marriage Compatibility Prediction using Machine Learning

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Create a sample dataset
data = {
    "age_difference": [2, 5, 1, 7, 3, 6, 4, 2, 8, 1],
    "same_caste": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    "lifestyle_match": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    "family_support": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    "emotional_score": [8, 4, 9, 3, 7, 5, 6, 8, 2, 9],
    "compatible": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Step 3: Basic data exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Visualization
sns.countplot(x="compatible", data=df)
plt.title("Compatibility Distribution")
plt.show()

# Step 5: Split data
X = df.drop("compatible", axis=1)
y = df["compatible"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

# Step 7: Train Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Classification Report
print(classification_report(y_test, dt_pred))

# Final Output
print("Model training completed successfully.")
