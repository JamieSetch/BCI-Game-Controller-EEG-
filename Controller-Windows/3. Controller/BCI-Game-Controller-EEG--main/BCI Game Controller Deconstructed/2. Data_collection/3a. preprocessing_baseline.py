import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# === Settings ===
data_dir = 'eeg_training_data'                                        # Directory containing EEG data
baseline_file = os.path.join(data_dir, 'baseline.csv')                # Path to baseline file
sample_rate = 250                                                     # Sampling rate in Hz
epoch_length = 250                                                    # Length of each epoch (1 second = 250 samples)

# === Frequency Bands (Hz) ===
bands = {                                                             # Defines EEG frequency bands
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Butterworth Bandpass Filter ===
def bandpass_filter(data, low, high, fs, order=3):                    # Apply bandpass filter to input data
    nyq = 0.5 * fs                                                    # Calculate Nyquist frequency
    low /= nyq                                                        # Normalize low cutoff
    high /= nyq                                                       # Normalize high cutoff
    b, a = butter(order, [low, high], btype='band')                   # Get filter coefficients
    return filtfilt(b, a, data, axis=1)                               # Apply zero-phase filter to data

# === Feature Extraction using Welch ===
def extract_bandpower_features(epoch, fs):                            # Extract average power in each band for each channel
    features = []
    for ch_data in epoch:                                             # For each EEG channel in the epoch
        freqs, psd = welch(ch_data, fs, nperseg=fs)                   # Compute Power Spectral Density using Welch
        for band, (low, high) in bands.items():                       # Loop through each defined frequency band
            idx = np.logical_and(freqs >= low, freqs <= high)         # Find indices within band
            features.append(np.mean(psd[idx]))                        # Append mean power in band
    return features

# === Process Baseline File ===
print("🔍 Processing baseline EEG...")                                # Notify start of baseline processing
baseline_epochs = []                                                  # Store extracted baseline feature vectors

try:
    df_base = pd.read_csv(baseline_file)                              # Load baseline CSV
    signal_base = df_base.drop(columns='label', errors='ignore').values.T  # Extract EEG data, transpose to shape (channels, samples)
    filtered_base = bandpass_filter(signal_base, 1, 50, sample_rate)  # Apply 1–50 Hz bandpass filter

    for i in range(0, filtered_base.shape[1] - epoch_length + 1, epoch_length):  # Segment data into 1s epochs
        segment = filtered_base[:, i:i + epoch_length]                # Extract segment
        features = extract_bandpower_features(segment, sample_rate)  # Extract bandpower features
        baseline_epochs.append(features)                              # Append to baseline list

    baseline_mean = np.mean(baseline_epochs, axis=0)                  # Compute average baseline feature vector
    print(f"✅ Baseline computed from {len(baseline_epochs)} epochs.")# Report success
except Exception as e:
    print(f"❌ Error processing baseline: {e}")                        # Handle and report error
    baseline_mean = None                                              # Fallback if baseline fails

# === Load and Process Trial Files ===
print("\n📁 Processing trial files...")                                # Notify start of trial processing
file_list = glob.glob(os.path.join(data_dir, 'trial_*.csv'))         # Get list of all trial CSV files
epochs, labels = [], []                                              # Initialize lists to hold features and labels

for file in file_list:
    df = pd.read_csv(file)                                           # Load trial file
    if df.shape[0] < 34:                                             # Skip files with too few rows
        print(f"⚠️ Skipping {file} (only {df.shape[0]} rows — too short)")
        continue

    label = df['label'].iloc[0]                                      # Get the label from the first row
    signal = df.drop(columns='label').values.T                       # Extract EEG signal, transpose

    try:
        filtered = bandpass_filter(signal, 1, 50, sample_rate)       # Apply bandpass filter to trial data
    except Exception as e:
        print(f"❌ Error filtering {file}: {e}")                     # Report filtering error
        continue

    for i in range(0, filtered.shape[1] - epoch_length + 1, epoch_length):  # Segment into 1s windows
        segment = filtered[:, i:i + epoch_length]                    # Extract segment
        features = extract_bandpower_features(segment, sample_rate)  # Extract features

        # Subtract baseline if available
        if baseline_mean is not None:
            features = np.array(features) - baseline_mean            # Normalize features using baseline

        epochs.append(features)                                      # Add features to dataset
        labels.append(label)                                         # Add corresponding label

print(f"\n✅ Processed {len(epochs)} valid epochs from {len(file_list)} trial files.")  # Summary

# === Convert to Arrays ===
X = np.array(epochs)                                                 # Convert features list to numpy array
y = np.array(labels)                                                 # Convert labels list to numpy array

# === Standardize Features ===
scaler = StandardScaler()                                            # Create standard scaler
X = scaler.fit_transform(X)                                          # Fit and transform features

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

# === Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)      # Create random forest model
clf.fit(X_train, y_train)                                            # Train classifier
y_pred = clf.predict(X_test)                                         # Predict on test data

# === Evaluation ===
print("\n📊 Classification Report:")                                 # Print classification results
print(classification_report(y_test, y_pred))                         # Show metrics: precision, recall, etc.

# === Save Model and Scaler ===
joblib.dump(scaler, 'scaler.pkl')                                    # Save the standardizer to file
joblib.dump(clf, 'classifier.pkl')                                   # Save the classifier to file
print("\n💾 Model and scaler saved.")                                # Confirm save
