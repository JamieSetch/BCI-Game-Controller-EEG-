import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.utils import to_categorical

DATA_DIR = # Adjust path as needed

WINDOW_SIZE = 250  # Size of each data window in samples (rows) / 250 samples typically represent 1 second of EEG data @ 250 Hz sampling
MAX_FULL_SAMPLE_SIZE = 70000  # max rows to load as single sample / 70,000 rows = roughly 280 seconds of sampling (our baseline data records just shy of this)

def load_data(data_dir, window_size=WINDOW_SIZE, max_full_sample_size=MAX_FULL_SAMPLE_SIZE):
    X = []  # Initialize list to hold EEG data samples.
    y = []  # Initialize list to hold corresponding labels for each sample.

    for file_name in os.listdir(data_dir):  # Iterate over every file in the specified data directory.
        if not file_name.endswith('.csv'):  # Skip files that do not have a '.csv' extension.
            continue  # Skip to the next file.

        file_path = os.path.join(data_dir, file_name)  # Create full file path by joining directory and filename.

        try:
            df = pd.read_csv(file_path)  # Attempt to read the CSV file into a pandas DataFrame.
        except Exception as e:
            print(f"Error loading {file_name}: {e}")  # Print error message if file loading fails.
            continue  # Skip to the next file.

        if 'label' not in df.columns:  # Check if the DataFrame contains a 'label' column.
            print(f"Warning: {file_name} missing 'label' column, skipping.")  # Warn if 'label' column is missing.
            continue  # Skip to the next file.

        data = df.drop(columns=['label']).values  # Extract EEG data values, excluding the 'label' column.
        labels = df['label'].values  # Extract the label values from the DataFrame.
        n_rows = data.shape[0]  # Get the number of rows (samples) in the EEG data.

        if n_rows <= max_full_sample_size:  # If the number of rows is less than or equal to max allowed for a single sample:
            X.append(data)  # Append the entire data array as one sample.
            y.append(labels[0])  # assume label consistent in file, append the first label value.
        else:  # If the file is too large to be one sample, split it into windows:
            num_windows = n_rows // window_size  # Calculate how many full windows of window_size fit in the data.

            if num_windows == 0:  # If the data is smaller than window size but larger than max_full_sample_size (rare edge case):
                print(f"Warning: {file_name} too short to split, skipping.")  # Warn that the file is too short to split properly.
                continue  # Skip this file.

            for i in range(num_windows):  # Loop over each window index:
                start = i * window_size  # Calculate start index of the current window.
                end = start + window_size  # Calculate end index of the current window.
                X.append(data[start:end])  # Append the data window slice as a separate sample.
                y.append(labels[start])  # label for window assumed constant, append label corresponding to start of window.

    X = np.array(X)  # Convert list of data samples to a NumPy array.
    y = np.array(y)  # Convert list of labels to a NumPy array.

    print(f"✅ Loaded {len(X)} samples with shape {X[0].shape if len(X) > 0 else 'N/A'}")  # Print confirmation of how many samples loaded and their shape.
    return X, y  # Return the data samples and labels as arrays.


def preprocess_data(X, y):
    X = X.astype('float32')  # Convert input data array to float32 type for consistency and compatibility with TensorFlow.
    X = np.expand_dims(X, axis=-1)  # Add channel dimension required by Conv2D layers: shape becomes (samples, height, width, channels).
    
    label_encoder = LabelEncoder()  # Initialize a label encoder to convert string labels into integer indices.
    y_encoded = label_encoder.fit_transform(y)  # Fit and transform labels from strings to integers.
    
    y_categorical = to_categorical(y_encoded)  # Convert integer encoded labels into one-hot encoded categorical vectors.
    
    return X, y_categorical, label_encoder  # Return processed data, one-hot labels, and encoder.


def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        InputLayer(input_shape=input_shape),  # Input layer specifying input sample shape.
        Conv2D(32, (3, 3), activation='relu', padding='same'),  # Conv2D with 32 filters, 3x3 kernel, ReLU activation, same padding.
        BatchNormalization(),  # Batch normalization for training stability.
        MaxPooling2D((2, 2)),  # Max pooling to downsample by factor of 2.
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Conv2D with 64 filters, 3x3 kernel, ReLU, same padding.
        BatchNormalization(),  # Batch normalization.
        MaxPooling2D((2, 2)),  # Another max pooling layer.
        Flatten(),  # Flatten 2D feature maps to 1D vector.
        Dense(128, activation='relu'),  # Fully connected dense layer with 128 neurons and ReLU.
        Dropout(0.5),  # Dropout with 50% rate to reduce overfitting.
        Dense(num_classes, activation='softmax')  # Output layer with softmax for classification.
    ])
    
    model.compile(optimizer='adam',  # Compile with Adam optimizer.
                  loss='categorical_crossentropy',  # Use categorical crossentropy loss for multi-class classification.
                  metrics=['accuracy'])  # Track accuracy metric.
    
    return model  # Return the compiled CNN model.


if __name__ == '__main__':
    print("🔍 Loading data...")  # Inform user data loading is starting.
    X, y = load_data(DATA_DIR)  # Load data and labels from data directory.

    if len(X) == 0:  # Check if any data was loaded.
        raise ValueError("No data loaded. Check your data directory and files.")  # Raise error if none.

    print("🔍 Preprocessing data...")  # Inform user preprocessing is starting.
    X_processed, y_processed, label_encoder = preprocess_data(X, y)  # Preprocess data: normalize, reshape, encode.

    print(f"Data reshaped to {X_processed.shape}")  # Print processed data shape.
    print(f"Number of classes: {y_processed.shape[1]}")  # Print number of output classes.

    X_train, X_test, y_train, y_test = train_test_split(  # Split data into train and test sets.
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed  # 80/20 split, stratified by label.
    )
    
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")  # Print train/test sample counts.

    model = build_cnn_model(input_shape=X_processed.shape[1:], num_classes=y_processed.shape[1])  # Build CNN model.

    print(model.summary())  # Print model architecture summary.

    print("🚀 Training model...")  # Notify training start.
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)  # Train model with 10% validation split.

    print("🧪 Evaluating model...")  # Notify evaluation start.
    loss, accuracy = model.evaluate(X_test, y_test)  # Evaluate model on test set.

    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")  # Print final test loss and accuracy.
