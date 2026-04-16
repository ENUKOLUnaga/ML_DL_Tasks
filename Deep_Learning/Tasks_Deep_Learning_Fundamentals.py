"""
#  Deep Learning Fundamentals using Loan Data

## Objective
This notebook introduces **Deep Learning concepts** using:
- A **Single-Layer Neural Network**
- A **Multi-Layer Neural Network (MLP)**

We apply them to a real **loan approval classification problem**.

Dataset:
 hdfc_loan_dataset_full_enriched.csv

##  What is Deep Learning?

Deep Learning is a subset of Machine Learning that uses
**Neural Networks with multiple layers** to learn complex patterns.

### Comparison
| Model | Capability |
|------|-----------|
| Linear / Logistic Regression | Linear relationships |
| Single-Layer NN | Slightly non-linear |
| Multi-Layer NN | Complex non-linear patterns |

Deep Learning is widely used in:
- Credit risk modeling
- Fraud detection
- Recommendation systems

##  Neural Network Basics

### Components:
- **Input Layer** → receives features
- **Hidden Layer(s)** → learns representations
- **Output Layer** → prediction

Each neuron performs:
Weighted Sum → Activation Function → Output
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/hdfc_loan_dataset_full_enriched - hdfc_loan_dataset_full_enriched.csv")
df.head()

num_features = [
    'Dependents', 'Applicant_Income', 'Coapplicant_Income',
    'Loan_Amount', 'Loan_Term_Months', 'Credit_History',
    'Age', 'CIBIL_Score', 'Annual_Household_Income',
    'Debt_to_Income_Ratio', 'Existing_EMIs',
    'Number_of_Previous_Loans', 'Default_History_Count',
    'Employment_Length_Years', 'Asset_Value',
    'Monthly_Expense', 'Loan_to_Annual_Income'
]

X = df[num_features]
y = df['Loan_Status']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

encoder.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""## 🔹 Single-Layer Neural Network

A single-layer neural network:
- Has **no hidden layers**
- Works similarly to Logistic Regression
- Learns linear decision boundaries

Used mainly for:
- Baseline deep learning models
- Comparison with deeper networks

"""

single_layer_model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(X_train_scaled.shape[1],))
])

single_layer_model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

single_layer_model.summary()

single_layer_model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

single_loss, single_acc = single_layer_model.evaluate(X_test_scaled, y_test)
print("Single-Layer NN Accuracy:", single_acc)

"""## 🔹 Multi-Layer Neural Network (MLP)

A Multi-Layer Perceptron:
- Has **one or more hidden layers**
- Learns complex non-linear patterns
- Forms the foundation of deep learning

This is the most common DL model for tabular data.

"""

mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

mlp_model.summary()

mlp_model.fit(
    X_train_scaled, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

mlp_loss, mlp_acc = mlp_model.evaluate(X_test_scaled, y_test)
print("Multi-Layer NN Accuracy:", mlp_acc)

"""##  Model Comparison

"""

pd.DataFrame({
    'Model': ['Single-Layer NN', 'Multi-Layer NN'],
    'Accuracy': [single_acc, mlp_acc]
})

"""## Business Interpretation

- Single-Layer NN ≈ Logistic Regression baseline
- Multi-Layer NN captures interactions between:
  - Income
  - Credit score
  - Debt ratio
- Better suited for complex loan approval decisions

 Deep learning requires:
- Proper scaling
- Careful tuning
- Explainability (next topic)

##  Practice Exercises

1. Increase the number of hidden layers and observe overfitting.
2. Change activation functions (ReLU → Tanh).
3. Reduce learning rate and compare convergence.
4. Compare DL accuracy with Random Forest from previous notebook.

"""

#1. Increase the number of hidden layers and observe overfitting.
deep_model = Sequential([
    Dense(128,activation = 'relu',input_shape = (X_train_scaled.shape[1],)),
    Dense(64,activation = 'relu'),
    Dense(32,activation = 'relu'),
    Dense(16,activation = 'relu'),
    Dense(1,activation = 'sigmoid')
])

deep_model.compile(
    optimizer = Adam(learning_rate = 0.01),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

deep_model.summary()

deep_model.fit(
    X_train_scaled, y_train,
    epochs = 30,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 1
)

deep_loss,deep_acc = deep_model.evaluate(X_test_scaled,y_test)
print("Multi-Layer NN Accuracy:", deep_acc)

#2. Change activation functions (ReLU → Tanh).
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tanh_model = Sequential([
    Dense(32, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(16, activation='tanh'),
    Dense(1, activation='sigmoid')
])

tanh_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_tanh = tanh_model.fit(
    X_train_scaled, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


#3. Reduce learning rate and compare convergence. (0.001->0.00001)
model_lr = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1,activation = 'sigmoid')
])

model_lr.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

histoy_lr = model_lr.fit(
    X_train_scaled, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#4. Compare DL accuracy with Random Forest from previous notebook.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=42,random_state=42)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy: ", rf_acc)

dl_pred = (tanh_model.predict(X_test_scaled) > 0.5).astype(int)
dl_acc = accuracy_score(y_test, dl_pred)
print("Deep Learning Accuracy:", dl_acc)