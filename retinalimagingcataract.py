# -*- coding: utf-8 -*-
"""RetinalImagingCataract.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RcL2xAM51kxxD4C_sXjhU1M73OiUvNkH

##**Mounting Google Drive**
"""

from google.colab import drive
drive.mount('/content/drive')

"""##**Importing Libraries**"""

pip install keras-tuner

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

#Hyperparameter Optimization
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

"""##**Determining File Path and Separating into Subfolders**"""

data_dir = '/content/drive/MyDrive/DerinOgrenme/RetinalImaging/Eyes/'

data_path = "/content/drive/MyDrive/DerinOgrenme/RetinalImaging/Eyes/"
cataract_path = os.path.join(data_path, "Cataract")
normal_path = os.path.join(data_path, "Normal")

print(os.listdir("/content/drive/MyDrive/DerinOgrenme/RetinalImaging/Eyes/"))

"""##**Specifying Image Size**"""

img_height, img_width = 128, 128

"""##**Loading Images**"""

def load_images(folder_path, label):
    images = []
    labels = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image {img_path}")
            continue
        img = cv2.resize(img, (img_height, img_width))
        images.append(img)
        labels.append(label)
        print(f"Image {idx + 1} loaded from {img_path}")
    return np.array(images), np.array(labels)

cataract_images, cataract_labels = load_images(cataract_path, 0)
normal_images, normal_labels = load_images(normal_path, 1)

"""##**Concatenating the Data Set as Cataract and Normal**"""

x = np.concatenate((cataract_images, normal_images), axis=0)
y = np.concatenate((cataract_labels, normal_labels), axis=0)

"""###Displaying the Number of Data Samples for X and Y"""

print("Number of data samples:", x.shape[0])
print("Number of data samples:", y.shape[0])

"""###Splitting the Data Set into Training (80%) and Test (20%) Sets"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 38)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.10, random_state = 38)

"""###Training and Finalizing the CNN Model"""

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_val = np.array(x_val)
y_val = np.array(y_val)

train_yCl = tf.keras.utils.to_categorical(y_train, num_classes=2)
test_yCl = tf.keras.utils.to_categorical(y_test, num_classes=2)
valid_yCl = tf.keras.utils.to_categorical(y_val, num_classes=2)

def build_model(hp):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=16),
                     kernel_size=(3, 3),
                     padding='same',
                     input_shape=(img_height, img_width, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout_conv1', min_value=0.2, max_value=0.5, step=0.1)))

    # Convolutional Layer 2
    model.add(Conv2D(filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=16),
                     kernel_size=(3, 3),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout_conv2', min_value=0.2, max_value=0.5, step=0.1)))

    # Flatten Layer
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(units=hp.Int('dense1_units', min_value=256, max_value=1024, step=32)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))

    # Dense Layer 2
    model.add(Dense(units=hp.Int('dense2_units', min_value=128, max_value=512, step=32)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))

    # Dense Layer 3
    model.add(Dense(units=hp.Int('dense3_units', min_value=64, max_value=256, step=16)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=hp.Float('dropout3', min_value=0.2, max_value=0.5, step=0.1)))

    # Output Layer
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Creating the tuner using RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Maximum number of models to try
    directory='my_dir',  # Directory to store saved models and results
    project_name='retinal_imaging_hyperparameter_tuning'  # Name of the tuning project
)

# Fitting the tuner with training data
tuner.search(x_train, train_yCl, epochs=10, validation_data=(x_val, valid_yCl))

# Getting the best hyperparameters and building the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Training the best model on the training set
best_model.fit(x_train, train_yCl, epochs=15, validation_data=(x_val, valid_yCl))

score_valid = best_model.evaluate(x_val, valid_yCl)
print("Validation Accuracy: ", score_valid[1])

score_test = best_model.evaluate(x_test, test_yCl)
print("Test Accuracy: ", score_test[1])

score_train = best_model.evaluate(x_train, train_yCl)
print("Training Accuracy: ", score_train[1])

# Making predictions on the test data
y_pred = best_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_yCl, axis=1)

# Calculating metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, labels=np.unique(y_true))

# Printing the results
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Displaying the Confusion Matrix
print('Confusion Matrix:')
print(conf_matrix)

# Showing the Classification Report
print('Classification Report:')
print(class_report)

"""###Splitting the Data Set into Training (65%) and Test (35%) Sets"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.35, random_state = 38)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.10, random_state = 38)

"""###Training and Finalizing the CNN Model"""

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_val = np.array(x_val)
y_val = np.array(y_val)

train_yCl = tf.keras.utils.to_categorical(y_train, num_classes=2)
test_yCl = tf.keras.utils.to_categorical(y_test, num_classes=2)
valid_yCl = tf.keras.utils.to_categorical(y_val, num_classes=2)

def build_model(hp):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=16),
                     kernel_size=(3, 3),
                     padding='same',
                     input_shape=(img_height, img_width, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout_conv1', min_value=0.2, max_value=0.5, step=0.1)))

    # Convolutional Layer 2
    model.add(Conv2D(filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=16),
                     kernel_size=(3, 3),
                     padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout_conv2', min_value=0.2, max_value=0.5, step=0.1)))

    # Flatten Layer
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(units=hp.Int('dense1_units', min_value=256, max_value=1024, step=32)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))

    # Dense Layer 2
    model.add(Dense(units=hp.Int('dense2_units', min_value=128, max_value=512, step=32)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))

    # Dense Layer 3
    model.add(Dense(units=hp.Int('dense3_units', min_value=64, max_value=256, step=16)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=hp.Float('dropout3', min_value=0.2, max_value=0.5, step=0.1)))

    # Output Layer
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Creating the tuner using RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Maximum number of models to try
    directory='my_dir',  # Directory to store saved models and results
    project_name='retinal_imaging_hyperparameter_tuning'  # Name of the tuning project
)

# Fitting the tuner with training data
tuner.search(x_train, train_yCl, epochs=10, validation_data=(x_val, valid_yCl))

# Getting the best hyperparameters and building the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Training the best model on the training set
best_model.fit(x_train, train_yCl, epochs=15, validation_data=(x_val, valid_yCl))

score_valid = best_model.evaluate(x_val, valid_yCl)
print("Validation Accuracy: ", score_valid[1])

score_test = best_model.evaluate(x_test, test_yCl)
print("Test Accuracy: ", score_test[1])

score_train = best_model.evaluate(x_train, train_yCl)
print("Training Accuracy: ", score_train[1])

# Making predictions on the test data
y_pred = best_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_yCl, axis=1)

# Calculating metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, labels=np.unique(y_true))

# Printing the results
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Displaying the Confusion Matrix
print('Confusion Matrix:')
print(conf_matrix)

# Showing the Classification Report
print('Classification Report:')
print(class_report)