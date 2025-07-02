import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, welch

# === Constants ===
SAMPLE_RATE = 250
WINDOW_SIZE = int(SAMPLE_RATE * 1.0)   # 1-second window
STEP_SIZE = int(SAMPLE_RATE * 0.5)     # 0.5-second step

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Filtering Function ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

# === Feature Extraction Function ===
def extract_bandpower_features(epoch, fs):
    features = []
    for ch_data in epoch:
        freqs, psd = welch(ch_data, fs, nperseg=fs)
        for low, high in BANDS.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.mean(psd[idx]))
    return features

# === Load Model and Scaler ===
try:
    scaler = joblib.load('scaler.pkl')
    clf = joblib.load('classifier.pkl')
    print("✅ Loaded model and scaler.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or scaler: {e}")

# === BrainFlow Setup ===
params = BrainFlowInputParams()

# ✅ SET THIS TO YOUR WINDOWS SERIAL PORT
# Example: 'COM3', 'COM4', etc. (check Device Manager if unsure)
params.serial_port = 'COMn'

board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

try:
    BoardShim.enable_dev_board_logger()
    board.prepare_session()
    board.start_stream()
    print("🧠 EEG stream started (press Ctrl+C to stop).")

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    buffer = np.empty((len(eeg_channels), 0))

    # === Live Loop ===
    while True:
        data = board.get_board_data()
        if data.shape[1] == 0:
            time.sleep(0.01)
            continue

        eeg_data = data[eeg_channels, :]
        buffer = np.hstack((buffer, eeg_data))

        while buffer.shape[1] >= WINDOW_SIZE:
            window = buffer[:, :WINDOW_SIZE]
            filtered = bandpass_filter(window, 1, 50, SAMPLE_RATE)
            features = extract_bandpower_features(filtered, SAMPLE_RATE)
            features_scaled = scaler.transform([features])
            prediction = clf.predict(features_scaled)
            print(f"🧠 Predicted Command: {prediction[0]}")

            buffer = buffer[:, STEP_SIZE:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🛑 Stopping EEG stream...")

finally:
    board.stop_stream()
    board.release_session()
    print("✅ EEG session ended.")
