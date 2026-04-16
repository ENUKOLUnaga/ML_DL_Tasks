"""
# Multinomial Logistic Regression using HDFC Loan Dataset

## Objective
This notebook demonstrates **Multinomial Logistic Regression** using a real-world
loan dataset.

### You will learn:
- What Multinomial Logistic Regression is
- When and why it is used
- How to convert loan data into a multiclass classification problem
- Model training, evaluation, and interpretation
- Practice through solved and unsolved exercises

## Logistic Regression Overview

Logistic Regression is a **supervised classification algorithm**.

### Types:
1. **Binary Logistic Regression**
   - Two classes (Approved / Rejected)
2. **Multinomial Logistic Regression**
   - More than two classes (Low / Medium / High Risk)
3. **Ordinal Logistic Regression**
   - Ordered classes

In this notebook, we focus on **Multinomial Logistic Regression**.

## When to Use Multinomial Logistic Regression

Use Multinomial Logistic Regression when:
- Target variable has **more than two categories**
- Categories are **mutually exclusive**
- No natural order is assumed

### Real-world examples:
- Loan risk classification: Low / Medium / High
- Customer segmentation: Basic / Premium / Enterprise
- Credit rating: Poor / Average / Good
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/hdfc_loan_dataset_full_enriched - hdfc_loan_dataset_full_enriched.csv")
df.head()

"""## Dataset Overview

This dataset contains enriched loan application data including:
- Applicant income
- Credit history / score
- Loan amount
- Loan term
- Employment details
- Demographic & financial indicators

We will **derive a multiclass target variable** to model loan risk.

"""

df.info()
df.describe()
df.isnull().sum()

"""## Creating a Multiclass Target

Multinomial Logistic Regression requires **3 or more classes**.

We will create a **Loan Risk Category** based on:
- CIBIL Score
- Debt-to-Income Ratio
- Loan Amount

Risk Classes:
- **Low Risk**
- **Medium Risk**
- **High Risk**

"""

def loan_risk(row):
    if row['CIBIL_Score'] >= 750 and row['Debt_to_Income_Ratio'] < 30:
        return 'Low_Risk'
    elif row['CIBIL_Score'] >= 650 and row['Debt_to_Income_Ratio'] < 45:
        return 'Medium_Risk'
    else:
        return 'High_Risk'

df['Loan_Risk'] = df.apply(loan_risk, axis=1)
df['Loan_Risk'].value_counts()

df['Loan_Risk'].value_counts().plot(kind='bar', title='Loan Risk Distribution')
plt.show()

"""## Feature Selection

We will use the following features:
- Applicant_Income
- Loan_Amount
- Loan_Term
- CIBIL_Score
- Debt_to_Income_Ratio
- Employment_Length_Years

Target variable:
 `Loan_Risk`

"""

features = [
    'Applicant_Income',
    'Loan_Amount',
    'Loan_Term_Months',
    'CIBIL_Score',
    'Debt_to_Income_Ratio',
    'Employment_Length_Years'
]

X = df[features]
y = df['Loan_Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

confusion_matrix(y_test, y_pred)

"""# UNSOLVED PRACTICE EXERCISES

## Exercise 1
Redefine loan risk categories using **Loan Amount** instead of Debt-to-Income.
Re-train the model and compare accuracy.
"""

#categorizing function
def categorize_loan_risk(row):
  if row['Loan_Amount']<1000000:
    return "Low"
  elif row['Loan_Amount']<2500000:
    return "Medium"
  else:
    return "High"

df['Loan_Risk'] = df.apply(categorize_loan_risk, axis=1)
df['Loan_Risk'].value_counts()

features = [
    'Applicant_Income',
    'Loan_Amount',
    'Loan_Term_Months',
    'CIBIL_Score',
    'Loan_Amount',
    'Employment_Length_Years'
]

X = df[features]
y = df['Loan_Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    multi_class = 'multinomial',
    solver = 'lbfgs',
    max_iter = 1000
)

model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report :\n",classification_report(y_test,y_pred))

"""## Exercise 2
Train the model using only:
- Credit_Score
- Debt_to_Income
- Employment_Years

Compare performance with the full-feature model.

"""

features = [
    'CIBIL_Score',
    'Debt_to_Income_Ratio',
    'Employment_Length_Years'
]

X = df[features]
y = df['Loan_Risk']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    multi_class = 'multinomial',
    solver = 'lbfgs',
    max_iter = 1000
)

model.fit(X_train_scaled, y_train)

y_predict = model.predict(X_test_scaled)

print("Accuracy : ",accuracy_score(y_test,y_predict))
print("\nClassification Report :\n",classification_report(y_test,y_pred))

"""## Exercise 3
Plot a confusion matrix heatmap and analyze:
- Which risk category is hardest to predict?
- Why?

"""

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

"""The confusion matrix heatmap was used to evaluate the classification performance of the model across different risk categories. By analyzing the matrix, it was observed that the medium-risk category is the hardest to predict, as it has the highest number of misclassifications compared to other classes. This is because medium-risk cases often share characteristics with both low-risk and high-risk categories, leading to overlapping feature distributions. As a result, the model finds it difficult to clearly distinguish this category, causing higher prediction errors.

## Exercise 4
Experiment with different solvers:
- `lbfgs`
- `newton-cg`
- `sag`

Compare convergence time and accuracy.
"""

features = [
    'Applicant_Income',
    'Loan_Amount',
    'Loan_Term_Months',
    'CIBIL_Score',
    'Loan_Amount',
    'Employment_Length_Years'
]

X = df[features]
y = df['Loan_Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import time
solvers = ['lbfgs', 'newton-cg', 'sag']
result = {}
for solver in solvers:
  start_time = time.time()

  model = LogisticRegression(
      multi_class = 'multinomial',
      solver = solver,
      max_iter = 1000
  )
  model.fit(X_train_scaled, y_train)

  y_pred = model.predict(X_test_scaled)

  accuracy = accuracy_score(y_test, y_pred)
  end_time = time.time()

  convergence_time= end_time - start_time

  result[solver] = (accuracy, convergence_time)
for solver,(accuracy, convergence_time) in result.items():
  print(f"{solver} -> Accuracy : {accuracy:.2f} convergence_time : {convergence_time:.2f}")

