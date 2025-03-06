import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Function to extract features (e.g., MFCC) from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    return mfcc_mean

# Load CREMA-D dataset and labels
def load_crema_dataset(data_dir):
    features, labels = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(data_dir, file)
            emotion = file.split('_')[2]  # Assuming filename format contains emotion
            features.append(extract_features(file_path))
            labels.append(emotion)

    return np.array(features), np.array(labels)

# Split dataset into train and test
def prepare_data(data_dir):
    features, labels = load_crema_dataset(data_dir)
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, num_classes=6)  # Assuming 6 emotion classes

    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Create a simple neural network model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(input_shape,), activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # Assuming 6 classes for emotions
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Feedback function
def provide_feedback(prediction, pause_feedback, fluency_feedback):
    emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust']
    predicted_emotion = emotions[np.argmax(prediction)]

    print(f"Predicted Emotion: {predicted_emotion}")
    print(f"Feedback on pauses: {pause_feedback}")
    print(f"Feedback on fluency: {fluency_feedback}")
    print(f"Overall feedback: Try to maintain a steady tone and limit unnecessary pauses.")

# Main function to train and test the model
def train_and_save_model(data_dir):
    X_train, X_test, y_train, y_test = prepare_data(data_dir)
    
    model = build_model(input_shape=X_train.shape[1])

    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # Save the model
    model.save('speech_model.h5')
    print("Model saved as speech_model.h5")

if __name__ == "__main__":
    data_dir = ''  # Replace with your dataset path
    train_and_save_model(data_dir)
