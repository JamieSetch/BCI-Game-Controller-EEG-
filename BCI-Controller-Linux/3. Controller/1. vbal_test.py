import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, welch
import os

# === Settings ===
sample_rate = 250
window_size = int(sample_rate * 1.0)   # 1-second window
step_size = int(sample_rate * 0.5)     # 0.5-second step

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Bandpass Filter Function ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

# === Bandpower Feature Extraction ===
def extract_bandpower_features(epoch, fs):
    features = []
    for ch_data in epoch:
        freqs, psd = welch(ch_data, fs, nperseg=fs)
        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.mean(psd[idx]))
    return features

# === Load Pretrained Model and Scaler ===
scaler = joblib.load('scaler.pkl')
clf = joblib.load('classifier.pkl')

# === BrainFlow Setup ===
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()

# 👇 Update your actual port here (check with `ls /dev/ttyUSB*` or `dmesg | grep tty`)
params.serial_port = '/dev/ttyUSB0'  # <-- Change to your OpenBCI Cyton port

board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

print("🔌 Connecting to OpenBCI board...")
board.prepare_session()
board.start_stream()
print("🧠 EEG stream started...")

eeg_channels = BoardShim.get_eeg_channels(board_id)
buffer = np.empty((len(eeg_channels), 0))

# === Main Prediction Loop ===
try:
    while True:
        data = board.get_board_data()
        if data.shape[1] == 0:
            time.sleep(0.01)
            continue

        eeg_data = data[eeg_channels, :]
        buffer = np.hstack((buffer, eeg_data))

        while buffer.shape[1] >= window_size:
            window = buffer[:, :window_size]
            filtered = bandpass_filter(window, 1, 50, sample_rate)
            features = extract_bandpower_features(filtered, sample_rate)
            features_scaled = scaler.transform([features])
            prediction = clf.predict(features_scaled)
            command = prediction[0]

            print(f"🔮 Predicted command: {command}")

            buffer = buffer[:, step_size:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("🛑 Stopping stream...")
    board.stop_stream()
    board.release_session()
    print("✅ Session closed.")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    board.stop_stream()
    board.release_session()
