import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Flatten, Dropout, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def create_model(hyperparameters):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(X_train.shape[1:])))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    for i in range(hyperparameters['num_layers'] - 1):
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Load tomographic data here
# X is your 3D ultrasound images and Y is the corresponding labels
# X, Y = load_data()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define some hyperparameters to try
hyperparameters_list = [{'num_layers': 2, 'learning_rate': 0.001}, 
                        {'num_layers': 3, 'learning_rate': 0.001}, 
                        {'num_layers': 2, 'learning_rate': 0.0001}]

# List to store the history of each training iteration
history_list = []

# For each set of hyperparameters
for hyperparameters in hyperparameters_list:
    model = create_model(hyperparameters)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=hyperparameters['learning_rate']), metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
    history_list.append(history)
