import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define a function to build a CNN model with variable hidden units
def build_model(hidden_units):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(hidden_units, activation='relu'),  # Variable hidden layer
        keras.layers.Dropout(0.5),  # Added dropout to prevent overfitting
        keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# List of hidden unit values to test
hidden_units_list = [32, 64, 128, 256, 512, 1024]

results = {}

for units in hidden_units_list:
    print(f"\nTraining model with {units} hidden units...")

    model = build_model(units)

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        epochs=30, validation_data=(x_test, y_test),
        batch_size=64, verbose=1,
        callbacks=[early_stopping]
    )

    # Store results
    results[units] = history.history

# Plot accuracy comparison
plt.figure(figsize=(10,5))
for units in hidden_units_list:
    plt.plot(results[units]['val_accuracy'], label=f"{units} hidden units")

plt.title('Validation Accuracy vs Hidden Units')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss comparison
plt.figure(figsize=(10,5))
for units in hidden_units_list:
    plt.plot(results[units]['val_loss'], label=f"{units} hidden units")

plt.title('Validation Loss vs Hidden Units')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()