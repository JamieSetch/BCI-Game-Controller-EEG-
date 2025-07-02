import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# === Settings ===
data_dir = 'eeg_training_data'
baseline_file = os.path.join(data_dir, 'baseline_eyes_open_recording.csv')  # Default file name
sample_rate = 250  # Hz
epoch_length = 250  # 1-second windows (250 samples)

# === Frequency Bands (Hz) ===
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Butterworth Bandpass Filter ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

# === Feature Extraction using Welch ===
def extract_bandpower_features(epoch, fs):
    features = []
    for ch_data in epoch:
        freqs, psd = welch(ch_data, fs, nperseg=fs)
        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.mean(psd[idx]))
    return features

# === Process Baseline File ===
print("🔍 Processing baseline EEG...")
baseline_epochs = []

try:
    df_base = pd.read_csv(baseline_file)
    signal_base = df_base.drop(columns='label', errors='ignore').values.T
    filtered_base = bandpass_filter(signal_base, 1, 50, sample_rate)

    for i in range(0, filtered_base.shape[1] - epoch_length + 1, epoch_length):
        segment = filtered_base[:, i:i + epoch_length]
        features = extract_bandpower_features(segment, sample_rate)
        baseline_epochs.append(features)

    baseline_mean = np.mean(baseline_epochs, axis=0)
    print(f"✅ Baseline computed from {len(baseline_epochs)} epochs.")
except Exception as e:
    print(f"❌ Error processing baseline: {e}")
    baseline_mean = None

# === Load and Process Trial Files ===
print("\n📁 Processing trial files...")
file_list = glob.glob(os.path.join(data_dir, 'trial_*.csv'))
epochs, labels = [], []

for file in file_list:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"❌ Could not read {file}: {e}")
        continue

    if df.shape[0] < 34:
        print(f"⚠️ Skipping {file} (only {df.shape[0]} rows — too short)")
        continue

    label = df['label'].iloc[0]
    signal = df.drop(columns='label').values.T

    try:
        filtered = bandpass_filter(signal, 1, 50, sample_rate)
    except Exception as e:
        print(f"❌ Error filtering {file}: {e}")
        continue

    for i in range(0, filtered.shape[1] - epoch_length + 1, epoch_length):
        segment = filtered[:, i:i + epoch_length]
        features = extract_bandpower_features(segment, sample_rate)

        # Subtract baseline if available
        if baseline_mean is not None:
            features = np.array(features) - baseline_mean

        epochs.append(features)
        labels.append(label)

print(f"\n✅ Processed {len(epochs)} valid epochs from {len(file_list)} trial files.")

# === Convert to Arrays ===
X = np.array(epochs)
y = np.array(labels)

# === Encode Labels (if strings) ===
le = LabelEncoder()
y = le.fit_transform(y)

# === Standardize Features ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === Evaluation ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save Model and Scaler ===
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(clf, os.path.join(model_dir, 'classifier.pkl'))
joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
print("\n💾 Model, scaler, and label encoder saved in:", os.path.abspath(model_dir))
