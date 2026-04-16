"""
# Introduction to Machine Learning

Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to **learn from data** and improve their performance over time without being explicitly programmed.

### Why Machine Learning?
- Automates decision-making using data.
- Helps find patterns and insights hidden in large datasets.
- Used in real-world applications like:
  - Email spam filtering
  - Fraud detection
  - Recommendation systems
  - Predictive analytics

# Machine Learning Types

Machine Learning (ML) can be broadly classified into **three main categories**:

---

## 1. Supervised Learning
Supervised learning works with **labeled data**, meaning the model learns the relationship between **input (X)** and **output (Y)**.  
The goal is to **predict outcomes** for new, unseen data.

### Common Types:
- **Regression:** Predict continuous values  
  - Examples: Predicting house prices, stock values, temperature  
  - Algorithms: Linear Regression, Ridge Regression, Lasso, SVR, Random Forest Regressor  

- **Classification:** Predict discrete categories  
  - Examples: Spam detection, sentiment analysis, fraud detection  
  - Algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, KNN  

### Real-world Examples:
| Task | Input | Output (Prediction) | Type |
|------|--------|--------------------|------|
| Predicting car price | Car features (age, mileage, model) | Price in $ | Regression |
| Email spam filter | Email text | Spam / Not Spam | Classification |
| Credit risk analysis | Customer data | Default / No Default | Classification |

---

## 2. Unsupervised Learning
Unsupervised learning works with **unlabeled data** — no predefined outputs are given.  
The goal is to **find hidden patterns, structure, or relationships** within the data.

### Common Types:
- **Clustering:** Group similar data points together  
  - Examples: Customer segmentation, document grouping, anomaly detection  
  - Algorithms: K-Means, Hierarchical Clustering, DBSCAN  

- **Dimensionality Reduction:** Reduce number of features while preserving information  
  - Examples: Visualizing high-dimensional data, speeding up ML models  
  - Algorithms: PCA (Principal Component Analysis), t-SNE  

### Real-world Examples:
| Task | Input | Output | Type |
|------|--------|--------|------|
| Customer segmentation | Purchase history | Customer groups | Clustering |
| Topic modeling | Text documents | Topic categories | Clustering |
| Image compression | Image pixels | Compressed image | Dimensionality Reduction |

---

## 3. Reinforcement Learning
Reinforcement Learning (RL) involves **an agent** that learns by interacting with its environment.  
The agent receives **rewards or penalties** for its actions and aims to **maximize total reward** over time.

### Key Terms:
- **Agent:** Learner/decision-maker  
- **Environment:** Where the agent operates  
- **Action:** What the agent can do  
- **Reward:** Feedback for actions  

### Common Algorithms:
- Q-Learning  
- Deep Q-Network (DQN)  
- Policy Gradient Methods  

### Real-world Examples:
| Task | Agent | Environment | Goal |
|------|--------|--------------|------|
| Game playing (e.g., Chess, Go) | AI model | Game rules | Win the game |
| Self-driving car | Vehicle control system | Roads, traffic | Reach destination safely |
| Robot navigation | Robot | Physical space | Optimize movement path |

---

### Quick Summary

| Learning Type | Data Type | Goal | Common Algorithms | Example |
|----------------|-----------|------|-------------------|----------|
| Supervised | Labeled | Predict outputs | Linear/Logistic Regression, Random Forest | Predict sales |
| Unsupervised | Unlabeled | Discover patterns | K-Means, PCA | Customer clustering |
| Reinforcement | Reward-based | Learn best actions | Q-Learning, DQN | Game AI |

---

## Example without using any model
"""

import matplotlib.pyplot as plt
import numpy as np

# Simulating labeled data (supervised)
x = np.linspace(0, 10, 50)
y = 2*x + np.random.randn(50)
plt.scatter(x, y, color='blue', label='Supervised (Labeled)')
plt.title("Supervised Learning Example: Regression Data")
plt.legend()
plt.show()

"""# Dataset Introduction

# The California Housing Dataset

The **California Housing dataset** contains information from the 1990 U.S. Census.  
Each record represents a block group (a small geographic area) and includes data such as:
- Median income
- House age
- Average number of rooms
- Population
- Latitude & Longitude
- Median house value (target variable)

We'll use this dataset to understand **Supervised Learning and Data Preprocessing** using the library called ***scikit learn***.

Use this reference to understand the scikit-learn library - [Link](https://https://scikit-learn.org/0.21/documentation.html)
"""

from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Display dataset info
df.head()

"""# Identifying the Problem Type

Here, our goal is to **predict house prices (median_house_value)** based on other features like income, rooms, and location.

 This is a **Supervised Learning** problem.

 The output variable (target) is **continuous**, so it’s a **Regression** task.
"""

df.info()
df.describe()

"""# Data Preprocessing Overview

Data preprocessing ensures our dataset is **clean, consistent, and ready for ML models**.

Key preprocessing steps:
1. Handling missing values  
2. Encoding categorical variables  
3. Feature scaling  
4. Splitting dataset into train/test sets

### Step 1: Checking any missing value
"""

df.isnull().sum()

"""### Step 2 : Feature and Target Seperation"""

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

"""### Step 3 : Train-Test Split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

"""### Step 4 : Feature Scaling"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Cross-Validation

Cross-validation helps evaluate how well a model generalizes to unseen data.

- Instead of a single train/test split, data is divided into multiple folds.
- The model is trained and validated multiple times, and the average performance is taken.

**K-Fold Cross-Validation:** The dataset is split into *k* equal parts (folds).
- Model is trained on k-1 folds and tested on the remaining fold.
- Repeated until all folds have been used once for testing.

"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model = LinearRegression()
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print("Cross-validation scores:", scores)
print("Average R2:", scores.mean())

"""# Practice Problem (Solved)

**Task:** Handle missing data and scale features for the given dataset.

We’ll simulate missing data in our dataset and demonstrate how to handle it.

"""

import numpy as np

# Simulate missing values in 2 columns
df_missing = df.copy()
df_missing.loc[df_missing.sample(frac=0.05).index, 'AveRooms'] = np.nan
df_missing.loc[df_missing.sample(frac=0.05).index, 'Population'] = np.nan

# Fill missing values using mean imputation
df_missing.fillna(df_missing.mean(), inplace=True)
df_missing.isnull().sum().head()

"""# Exercise for you

1. Identify whether each of these problems is **supervised**, **unsupervised**, or **reinforcement**:
   - Predicting student exam scores.
   - Clustering customers based on purchase patterns.
   - Training an agent to play chess.

2. Load the `fetch_california_housing()` dataset again:
   - Perform EDA (check correlations, histograms, and summary stats).
   - Normalize the data using `MinMaxScaler` instead of `StandardScaler`.
   - Explain how scaling affects the dataset.

3. Split the dataset using different test sizes (0.1, 0.3, 0.4) and observe how model performance changes.

4. Use K-Fold cross-validation with `k=10` and compare the results with `k=5`.

5. Discuss: Why is cross-validation preferred over a single train-test split?

1.Identify whether each of these problems is supervised, unsupervised, or reinforcement:


1.   Predicting student exam scores.
2.   Clustering customers based on purchase patterns.
3.   Training an agent to play chess


**1. Predicting student exam scores**                                     
Type : Supervised Learning Algorithm                                             
Supervised Learning Algorithm because the model is trained using labeled data.   
The input features such as study hours, attendance are provided along with corresponding output exam scores. The model learns the relationship between inputs and outputs to predict scores for new data.

**2. Clustering customers based on purchase patterns**                       
Type : Unsupervised Learning Algorithm                                    
Unsupervised Learning Algorithm because the model is trained on unlabeled data.
There are no predefined labels or target outputs. The model analyzes customer purchase data and groups based on similarities in behavior.                       

**3. Training an agent to play chess**                                             
Type : Reinforcement Learning Algorithm                                           
Reinforcement learning, where an agent learns by interacting with an environment (the chessboard). The agent takes actions (moves) and receives feedback in the form of rewards or penalties (winning or losing).
"""

"""2. Load the `fetch_california_housing()` dataset again:
   - Perform EDA (check correlations, histograms, and summary stats).
   - Normalize the data using `MinMaxScaler` instead of `StandardScaler`.
   - Explain how scaling affects the dataset.
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd

#Load the dataset
california = fetch_california_housing(as_frame = True)
df = california.frame

print(df.head())

df.describe()

df.info()

df.isnull().sum()

import matplotlib.pyplot as plt
df.hist(figsize=(12,10), bins=30)
plt.show()

df.corr()['MedHouseVal'].sort_values(ascending=False)

from sklearn.model_selection import train_test_split

x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import MinMaxScaler

minmaxscaler = MinMaxScaler()

x_train_scaled = minmaxscaler.fit_transform(x_train)
x_test_scaled = minmaxscaler.transform(x_test)

print("Training samples : ",x_train_scaled.shape[0])
print("Testing samples : ",x_test_scaled.shape[0])

"""3. Split the dataset using different test sizes (0.1, 0.3, 0.4) and observe how model performance changes."""

x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

result ={}

test_sizes = [0.1,0.3,0.4]

for size in test_sizes:
  x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = size, random_state = 42)

  model = LinearRegression()
  model.fit(x_train, y_train)

  y_pred = model.predict(x_test)

  r2 = r2_score(y_test, y_pred)

  result[size] = r2

for size, r2 in result.items():
  print(f"Test Size : {size}, r2_score : {r2:.4f}")

"""4. Use K-Fold cross-validation with k=10 and compare the results with k=5"""

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import MinMaxScaler

minmaxscaler = MinMaxScaler()

X_train_scaled = minmaxscaler.fit_transform(X_train)
X_test_scaled = minmaxscaler.transform(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()

scores_k5 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
scores_k10 = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='r2')

print("K=5 Average:", scores_k5.mean())
print("K=10 Average:", scores_k10.mean())

"""5. Discuss: Why is cross-validation preferred over a single train-test split?     
  Cross-validation is preferred over a single train-test split because it provides a more reliable and unbiased estimate of model performance. Unlike a single split, it evaluates the model on multiple subsets of data, reducing dependency on one particular division. It also ensures better utilization of the dataset, as every data point is used for both training and testing. Overall, cross-validation produces more stable and accurate results, helping the model generalize better to unseen data.

"""