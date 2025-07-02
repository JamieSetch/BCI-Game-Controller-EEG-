import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.utils import to_categorical

# === Linux-specific path — change this as needed ===
DATA_DIR = '/home/yourusername/eeg_training_data'  # 👈 Set your actual path

WINDOW_SIZE = 250
MAX_FULL_SAMPLE_SIZE = 70000  # max rows to load as single sample

def load_data(data_dir, window_size=WINDOW_SIZE, max_full_sample_size=MAX_FULL_SAMPLE_SIZE):
    X = []
    y = []
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.csv'):
            continue
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Error loading {file_name}: {e}")
            continue

        if 'label' not in df.columns:
            print(f"⚠️ Warning: {file_name} missing 'label' column, skipping.")
            continue

        data = df.drop(columns=['label']).values
        labels = df['label'].values

        n_rows = data.shape[0]

        if n_rows <= max_full_sample_size:
            X.append(data)
            y.append(labels[0])  # Assume consistent label
        else:
            num_windows = n_rows // window_size
            if num_windows == 0:
                print(f"⚠️ Warning: {file_name} too short to split, skipping.")
                continue
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                X.append(data[start:end])
                y.append(labels[start])

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Loaded {len(X)} samples with shape {X[0].shape if len(X) > 0 else 'N/A'}")
    return X, y

def preprocess_data(X, y):
    X = X.astype('float32')

    # Add channel dimension for Conv2D input
    X = np.expand_dims(X, axis=-1)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # One-hot encode
    y_categorical = to_categorical(y_encoded)

    return X, y_categorical, label_encoder

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("🔍 Loading data from:", os.path.abspath(DATA_DIR))
    X, y = load_data(DATA_DIR)

    if len(X) == 0:
        raise ValueError("No data loaded. Check your Linux path and files.")

    print("🔄 Preprocessing data...")
    X_processed, y_processed, label_encoder = preprocess_data(X, y)

    print(f"📐 Input shape: {X_processed.shape}")
    print(f"🧾 Number of classes: {y_processed.shape[1]}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )

    print(f"📊 Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Build model
    model = build_cnn_model(input_shape=X_processed.shape[1:], num_classes=y_processed.shape[1])
    print("🧠 Model Summary:")
    model.summary()

    # Train model
    print("🚀 Training model...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

    # Evaluate model
    print("🧪 Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"✅ Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
