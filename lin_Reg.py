import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('dataset_1.csv')

# Heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of the Dataset')
plt.show()

# Min-Max scaling
X = df.drop('Unemployment', axis=1)
y = df['Unemployment']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Output the regression score, coefficients, intercept, and mean squared error
print("Regression Score (R^2):", lr_model.score(X_test, y_test))
print("Coefficients:", lr_model.coef_)
print("Intercept:", lr_model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# Scatter plot for true output vs predicted output
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='True Output')
plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.5, label='Predicted Output')
plt.title('True vs Predicted Unemployment Rates')
plt.xlabel('Test Cases')
plt.ylabel('Unemployment Rate')
plt.legend()
plt.show()
