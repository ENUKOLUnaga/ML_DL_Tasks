"""
# Decision Trees and Random Forest - Loan Risk Modeling

** TARGET & FEATURE CLARITY**

Loan_Status   →  Approved / Rejected / Pending (or similar)

#  Decision Trees, Random Forest & XGBoost
## Loan Status Prediction using HDFC Loan Dataset

### Objective
- Build tree-based classification models
- Predict loan approval status
- Compare Decision Tree, Random Forest, and XGBoost
- Understand feature importance for business decisions

##  Tree-Based Models Overview

### Decision Tree
- Rule-based splits
- Easy to interpret
- Can overfit

### Random Forest
- Ensemble of decision trees
- Reduces variance
- More stable predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/hdfc_loan_dataset_full_enriched - hdfc_loan_dataset_full_enriched.csv")
df.head()

"""## Dataset Overview

The dataset contains:
- Applicant demographics
- Financial indicators
- Credit behavior
- Geographic and institutional data
- Textual fields (excluded from modeling)

Target variable:
 `Loan_Status`

"""

drop_cols = [
    'Loan_ID', 'Customer_Name', 'Phone_Number', 'Email',
    'Aadhaar_Synthetic', 'Application_Text',
    'Customer_Feedback', 'Agent_Notes', 'Unnamed: 39', 'Bank'
]

df_model = df.drop(columns=drop_cols, errors='ignore')

# Fill missing categorical values
df_model['Business_Type'].fillna('Unknown', inplace=True)
df_model['Co-signer_Relationship'].fillna('None', inplace=True)

df_model.isnull().sum()

target = 'Loan_Status'

X = df_model.drop(columns=[target])
y = df_model[target]

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

num_features, cat_features

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

"""# ** DECISION TREE**"""

dt_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', DecisionTreeClassifier(max_depth=6, random_state=42))
])

dt_pipeline.fit(X_train, y_train)

dt_pred = dt_pipeline.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

"""# ** RANDOM FOREST**"""

rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

"""# ** MODEL COMPARISON**"""

pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ]
})

"""# **UNSOLVED EXERCISE**

 Exercise 1 (Unsolved)
 Feature Importance Analysis
Task

Extract feature importance from:

1. Random Forest

2. Decision Tree

Identify the top 5 most important features.

Answer:

Which features influence loan approval the most?

Do both models agree on the top features?
"""

#Random Forest importance
feature_names = rf_pipeline.named_steps['preprocess'].get_feature_names_out()
rf_model = rf_pipeline.named_steps['model']
rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Decision Tree importance
feature_names1 = dt_pipeline.named_steps['preprocess'].get_feature_names_out()
dt_model = dt_pipeline.named_steps['model']
dt_importance = pd.DataFrame({
    'Feature': feature_names1,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Random Forest Top 5:\n", rf_importance.head())
print("\nDecision Tree Top 5:\n", dt_importance.head())

"""Feature importance was extracted using both Random Forest and Decision Tree models. The top features influencing loan approval were identified as Credit Score, Income, Debt-to-Income Ratio, Employment Years, and Loan Amount. These features are critical as they directly reflect a borrower’s financial stability and repayment capacity."""

print(rf_pipeline.named_steps)

""" Exercise 2 (Unsolved)
 Model Tuning & Overfitting Check
Task

1. Train a Decision Tree with:

max_depth=None

2. Train another Decision Tree with:

max_depth=4

3. Compare:

Training accuracy

Test accuracy

4. Answer:

Which model overfits?

Why does depth control matter in loan risk models?
"""

df_pipeline_none = Pipeline([
    ('preprocess' , preprocessor),
    ('model' , DecisionTreeClassifier(max_depth=None, random_state=42))
])

df_pipeline_none.fit(X_train, y_train)


df_pipeline_4 = Pipeline([
    ('preprocess' , preprocessor),
    ('model' , DecisionTreeClassifier(max_depth = 4, random_state = 42))
])

df_pipeline_4.fit(X_train, y_train)

df_pred1_train = df_pipeline_none.predict(X_train)
print("Decision Tree Train Accuracy for max_depth None:", accuracy_score(y_train, df_pred1_train))
df_pred1_test = df_pipeline_none.predict(X_test)
print("Decision Tree Test Accuracy for max_depth None:", accuracy_score(y_test, df_pred1_test))
print(classification_report(y_test,df_pred1_test))

df_pred2_train = df_pipeline_4.predict(X_train)
print("Decision Tree Train Accuracy for max_depth 4:", accuracy_score(y_train, df_pred2_train))
df_pred2_test = df_pipeline_4.predict(X_test)
print("Decision Tree Accuracy for max_depth 4 :", accuracy_score(y_test, df_pred2_test))
print(classification_report(y_test,df_pred2_test))

"""The model with max_depth=None overfits because it learns noise and specific patterns in the training data rather than general trends.

Depth control is important in loan risk models because it prevents overfitting, improves generalization to new applicants, and ensures more reliable and interpretable decision-making in financial systems.
"""

