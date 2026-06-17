#✅ California Housing Dataset — Complete with Comments)
# --------------------------------------------------------
# 📌 1. Import all required libraries
# --------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#load the California Housing dataset
california_data = fetch_california_housing(as_frame=True)

# Convert to a pandas DataFrame for easy exploration
df = pd.DataFrame(california_data.data, columns=california_data.feature_names)
df['MedHouseValue'] = california_data.target #adds the target variable to the DataFrame

#Show the first few rows of the dataset
print(df.head())

# 📌 3. Quick data overview
# --------------------------------------------------------
# print(df.shape)      # Number of rows & columns
# print(df.dtypes)     # Data types of each column
# print(df.isnull().sum())  # Number of missing values in each column

print(df.info())     # Data types

print(df.describe())        # Statistical summary

# 📌 4. Exploratory Data Analysis (simple plots)
# --------------------------------------------------------

"""

# Histogram of the target (house prices)
plt.hist(df['MedHouseValue'], bins=40)
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.title("Distribution of House Prices")
plt.show()



# Heatmap to show correlation between features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

"""

# 📌 5. Split dataset for training and testing
# --------------------------------------------------------

X = df.drop("MedHouseValue", axis=1)  # Features
y = df["MedHouseValue"]               # Target (Regression)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape

# 📌 6. Scale the features (very important for regression)
# --------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 📌 7. Train a Linear Regression model
# --------------------------------------------------------

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Print the learned coefficients
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)