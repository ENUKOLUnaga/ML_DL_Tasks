"""
#  Convolutional Neural Networks (CNN) using CIFAR-10 (Automobile Images)

## Objective
This notebook explains **CNN fundamentals** by focusing on a **single visual class**
— *Automobile images* from CIFAR-10.

You will learn:
- CNN architecture & components
- Convolution, ReLU, Pooling
- Flattening & Fully Connected layers
- Softmax & Cross-Entropy loss
- TensorFlow 2.x & Keras
- OpenCV image handling
- TensorBoard visualization

Dataset:
 CIFAR-10 (Kaggle)
Class Used:
 Automobile (label = 1)

## What is a Convolutional Neural Network (CNN)?

CNNs are neural networks designed specifically for **image data**.

They automatically learn:
- Edges
- Shapes
- Object parts
- High-level visual concepts

CNNs are widely used in:
- Image classification
- Object detection
- Autonomous vehicles
- Medical imaging

##  CNN Core Components

1. **Convolution Operation**
   - Extracts features using filters (kernels)

2. **ReLU Layer**
   - Introduces non-linearity

3. **Pooling Layer**
   - Reduces spatial size (downsampling)

4. **Flattening**
   - Converts feature maps into vectors

5. **Fully Connected Layer**
   - Learns decision boundaries

6. **Softmax**
   - Converts scores to probabilities

7. **Cross-Entropy Loss**
   - Measures classification error

## Loading CIFAR-10 from Kaggle into Colab

We will use the **Kaggle API** to download CIFAR-10 directly into Colab.
"""

from google.colab import files
files.upload()
## Upload kaggle.json(Kaggle API)
"""
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c cifar-10

!unzip cifar-10.zip
!unzip train.7z
!unzip test.7z
"""

from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

"""# **Import the Libraries**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

print(X_train.shape)
print(X_test.shape)

"""## CIFAR-10 Class Labels

| Label | Class |
|-----|------|
| 0 | Airplane |
| 1 | **Automobile** |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

 We will **filter and use only Automobile images (label = 1)**.

# **LOAD IMAGES USING OpenCV**
"""

y_train = (y_train == 1).astype(int)
y_test = (y_test == 1).astype(int)

print(y_train.shape)
print(y_test.shape)

"""## Data Preprocessing

- Normalize pixel values
- Convert labels to categorical format

"""

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

"""## CNN Architecture for Automobile Classification

Architecture:
1. Conv2D → ReLU
2. MaxPooling
3. Conv2D → ReLU
4. MaxPooling
5. Flatten
6. Fully Connected Layer
7. Softmax Output

## **BUILD CNN**
"""

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

"""## **COMPILE CODE**"""

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

"""# **TensorBoard Setup**"""

from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir)

"""# **TRAIN MODEL**"""

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tensorboard_cb]
)

"""# **LAUNCH TENSORBOARD**"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs/fit

"""# **MODEL EVALUATION**"""

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Automobile Classification Accuracy:", test_acc)

"""## Softmax Output

Softmax converts raw scores into probabilities.

"""

predictions = model.predict(X_test[:5])
np.argmax(predictions, axis=1)

"""## Practice Exercises

1. Add Dropout layers and observe overfitting.
2. Increase convolution filters (32 → 64 → 128).
3. Convert images to grayscale using OpenCV.
4. Add Batch Normalization.
5. Compare accuracy with full CIFAR-10 (10 classes).

Bonus:
- Visualize feature maps from convolution layers.

"""

#1. Add Dropout layers and observe overfitting.
#2. Increase convolution filters (32 → 64 → 128).
from tensorflow.keras.layers import Dropout
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tensorboard_cb]
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Automobile Classification Accuracy:", test_acc)

print(X_train.shape)
print(X_test.shape)

#3. Convert images to grayscale using OpenCV.
import cv2
import numpy as np

def gray_to_rgb(images):
    rgb_images = np.array([
        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for img in images
    ])

    return rgb_images / 255.0

X_train = gray_to_rgb(X_train)
X_test = gray_to_rgb(X_test)

print(X_train.shape)
print(X_test.shape)

def convert_to_grayscale(images):
    # If already grayscale, return as is
    if images.shape[-1] == 1:
        return images

    # Convert each image to grayscale
    gray_images = np.array([
        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        for img in images
    ])

    # Reshape and normalize
    return gray_images.reshape(-1, images.shape[1], images.shape[2], 1) / 255.0


# Apply conversion
X_train = convert_to_grayscale(X_train)
X_test = convert_to_grayscale(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print(X_train.shape)
print(X_test.shape)

#4. Add Batch Normalization.
from tensorflow.keras.layers import Dropout, BatchNormalization
model_normalization = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model_normalization.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model_normalization.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tensorboard_cb]
)

"""**Compare accuracy with full CIFAR-10 (10 classes)**"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model_rgb = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_rgb.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_rgb.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

rgb_acc = model_rgb.evaluate(X_test, y_test)[1]
print("RGB Accuracy:", rgb_acc)

import numpy as np

X_train_gray = np.dot(X_train[...,:3], [0.299, 0.587, 0.114])
X_test_gray  = np.dot(X_test[...,:3],  [0.299, 0.587, 0.114])

# Reshape
X_train_gray = X_train_gray.reshape(-1,32,32,1)
X_test_gray  = X_test_gray.reshape(-1,32,32,1)

model_gray = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_gray.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_gray.fit(X_train_gray, y_train, epochs=10, validation_data=(X_test_gray, y_test))

gray_acc = model_gray.evaluate(X_test_gray, y_test)[1]
print("Grayscale Accuracy:", gray_acc)

print("RGB Accuracy:", rgb_acc)
print("Grayscale Accuracy:", gray_acc)

"""**Visualize feature maps from convolution layers**"""

model_rgb.predict(X_test[:1])

layer_outputs = [
    layer.output for layer in model_rgb.layers
    if 'conv' in layer.name
]

activation_model = model(inputs=model_rgb.inputs, outputs=layer_outputs)

import numpy as np

# Pick one image
img = X_test[0]

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Get feature maps
feature_maps = activation_model.predict(img)

import matplotlib.pyplot as plt

layer_names = [layer.name for layer in model_rgb.layers if 'conv' in layer.name]

for layer_name, feature_map in zip(layer_names, feature_maps):
    print("Layer:", layer_name)

    # Show first 6 feature maps
    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.imshow(feature_map[0, :, :, i], cmap='gray')
        plt.axis('off')

    plt.show()