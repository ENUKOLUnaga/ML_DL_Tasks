"""
# Introduction to Regression

Regression is a **supervised learning** technique used to model the relationship between a **dependent variable (target)** and one or more **independent variables (features)**.

### Types of Regression:
1. **Simple Linear Regression:**  
   - One independent variable (X)  
   - Equation:  Ŷ = b₀ + b₁X  
   - Example: Predicting temperature based on humidity

2. **Multiple Linear Regression:**  
   - Multiple independent variables (X₁, X₂, X₃, …)  
   - Equation: Ŷ = b₀ + b₁X₁ + b₂X₂ + … + bₙXₙ  
   - Example: Predicting temperature using humidity, wind speed, visibility, and pressure

We'll explore both types step by step using weather data. *[Dataset Link](https://https://drive.google.com/drive/folders/12-Yod1dZRdvMmPFZ8n8KO1jUZELXTRaw?usp=sharing)*

# Linear Regression with Weather Data

### Topics Covered:
1. Understanding Linear Regression  
2. Implementing Linear Regression *from scratch (no libraries)*  
3. Using Scikit-learn for Simple & Multiple Linear Regression  
4. Practice Exercises

## Understanding Linear Regression

Linear Regression tries to establish a **linear relationship** between a dependent variable (Y) and one or more independent variables (X).

The equation for simple linear regression:

\[
Y = mX + b
\]

Where:
- **m** = slope (change in Y for one unit change in X)
- **b** = intercept (Y value when X = 0)

The goal is to find *m* and *b* such that the prediction error (difference between actual and predicted Y) is minimized.

## Implementing Linear Regression from Scratch (No Library)

We'll compute slope (*m*) and intercept (*b*) using the formulas:

\[
m = \frac{n(\sum xy) - (\sum x)(\sum y)}{n(\sum x^2) - (\sum x)^2}
\]
\[
b = \frac{(\sum y) - m(\sum x)}{n}
\]
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load dataset
data = pd.read_csv("/content/drive/MyDrive/weatherHistory.csv")

# Use small subset for demo
X = data['Humidity'].values[:100]  # independent variable
Y = data['Apparent Temperature (C)'].values[:100]  # dependent variable

# Calculate slope (m) and intercept (b)
n = len(X)
m = (n*(X*Y).sum() - X.sum()*Y.sum()) / (n*(X**2).sum() - (X.sum())**2)
b = (Y.sum() - m*X.sum()) / n

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Predict for random X
def predict(x):
    return m*x + b

# Example prediction
x_test = 0.75
print(f"Predicted Apparent Temperature for Humidity={x_test}: {predict(x_test):.2f} °C")

import matplotlib.pyplot as plt

plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, predict(X), color='red', label='Regression Line')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature (C)')
plt.legend()
plt.title('Linear Regression from Scratch')
plt.show()

"""## 🧠 Practice 1
Implement the above formula using **NumPy vectorization** instead of manual loops.  
Try predicting the apparent temperature for humidity values `[0.5, 0.6, 0.8]`.

"""

# Write your code
import pandas as pd

# Load dataset
data = pd.read_csv("/content/drive/MyDrive/weatherHistory.csv")

# Use small subset for demo
X = data['Humidity'].values[:100]  # independent variable
Y = data['Apparent Temperature (C)'].values[:100]  # dependent variable

# Calculate slope (m) and intercept (b)
n = len(X)
m = (n*(X*Y).sum() - X.sum()*Y.sum()) / (n*(X**2).sum() - (X.sum())**2)
b = (Y.sum() - m*X.sum()) / n

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Predict for random X
def predict(x):
    return m*x + b

x_test = [0.5, 0.6, 0.8]

for test in x_test:
  print(f"Predicted Apparent Temperature for Humidity={test}: {predict(test):.2f} °C")

import matplotlib.pyplot as plt
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, predict(X), color='red', label='Regression Line')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature (C)')
plt.legend()
plt.title('Linear Regression from Scratch')
plt.show()

"""## Implementing Linear Regression using Scikit-Learn

Now that we understand the math, let’s use `LinearRegression` from sklearn to:
1. Train a model on humidity vs. apparent temperature  
2. Predict for new humidity values  
3. Compare results with our from-scratch implementation

"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Prepare data
X = data[['Humidity']]
y = data['Apparent Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature (C)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

"""## Practice 2
- Predict *Apparent Temperature* for humidity = 0.65 and 0.80.  
- Plot the regression line on the full dataset (not just test data).

"""

from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X = data[['Humidity']]
y = data['Apparent Temperature (C)']


model = LinearRegression()
model.fit(X, y)

humidity_values = np.array([0.65, 0.80])

predictions = model.predict(humidity_values.reshape(-1, 1))

print(f" Apparent Temperature :")
print(f"Humidity 0.65 -> {predictions[0]:.2f} °C")
print(f"Humidity 0.80 -> {predictions[1]:.2f} °C")

import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Actual Data')
y_pred_full = model.predict(X)
plt.plot(X, y_pred_full, color='red', label='Regression Line')
plt.xlabel('Humidity')
plt.ylabel('Apparent Temperature (C)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

"""# Multiple Linear Regression

In multiple regression, we use **multiple predictors** to predict a single target variable.

\[
Y = b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n
\]

We'll predict *Apparent Temperature (C)* using:
- Temperature (C)  
- Humidity  
- Wind Speed (km/h)  
- Visibility (km)  
- Pressure (millibars)

"""

features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']
X_multi = data[features]
y_multi = data['Apparent Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

y_pred_multi = multi_model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred_multi))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_multi)))

coeffs = pd.DataFrame(multi_model.coef_, features, columns=['Coefficient'])
coeffs

"""## Exercise 3
Predict *Apparent Temperature* when:

| Temperature (C) | Humidity | Wind Speed (km/h) | Visibility (km) | Pressure (millibars) |
|------------------|----------|------------------|-----------------|---------------------|
| 12 | 0.65 | 10 | 9 | 1015 |

"""

sample = np.array([[12, 0.65, 10, 9, 1015]])
predictions = multi_model.predict(sample)

print(f"Predicted Apparent Temperature : {predictions[0]: .2f} °C")

"""## Practice 4
1. Train your own model using any **subset of features** (e.g., only Humidity, Wind Speed, Pressure).  
2. Evaluate which combination gives the **highest R² score**.  
3. Interpret the meaning of coefficients — which features affect apparent temperature most?

"""

from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

y = data['Apparent Temperature (C)']

feature_sets ={
    'Humidity': ['Humidity'],
    'Wind Speed (km/h)': ['Wind Speed (km/h)'],
    'Pressure (millibars)': ['Pressure (millibars)'],
    'Humidity & Wind Speed (km/h)': ['Humidity', 'Wind Speed (km/h)'],
    'Humidity & Pressure (millibars)': ['Humidity', 'Pressure (millibars)'],
    'Wind Speed (km/h) & Pressure': ['Wind Speed (km/h)', 'Pressure (millibars)']
}

result = {}

for feature_name, features in feature_sets.items():
  X = data[features]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 42)
  model = LinearRegression()
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  result[feature_name] = r2_score(y_test,y_pred)

for feature_name, score in result.items():
  print(f"{feature_name} : {score}")

features = ['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']

X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")