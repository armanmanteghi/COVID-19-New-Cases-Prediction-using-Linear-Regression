import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV
file_path = r'C:\Users\arman\OneDrive\Desktop\Coding\country_wise_latest.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Check column names to ensure they match
print(df.columns)

# Feature and target variables
# Adjust these based on your actual CSV column names
X = df[['Confirmed last week']]
y = df['New cases']

# Handle any missing values or data types if necessary
X = X.fillna(0)  # Example: replace missing values with 0
y = y.fillna(0)

# Ensure data is numeric
X = pd.to_numeric(X.squeeze(), errors='coerce').fillna(0).values.reshape(-1, 1)
y = pd.to_numeric(y, errors='coerce').fillna(0).values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plotting
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Fitted line')
plt.xlabel('Confirmed cases last week')
plt.ylabel('New cases')
plt.title('COVID-19 New Cases Prediction')
plt.legend()
plt.show()

# Predicting future cases
future_confirmed_last_week = np.array([[80000]])  # Example future value
future_pred = model.predict(future_confirmed_last_week)
print(f'Predicted new cases for {future_confirmed_last_week[0][0]} confirmed cases last week: {future_pred[0]}')
