import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, welch

# === Settings ===
sample_rate = 250                              # Sampling rate in Hz
window_size = int(sample_rate * 1.0)           # 1 second window (250 samples)
step_size = int(sample_rate * 0.5)             # 0.5 seconds step (125 samples)

bands = {                                      # Frequency bands for feature extraction
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Filter & Feature Extraction ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs                              # Nyquist frequency
    b, a = butter(order, [low / nyq, high / nyq], btype='band')  # Bandpass filter coefficients
    return filtfilt(b, a, data, axis=1)        # Apply filter along time axis

def extract_bandpower_features(epoch, fs):
    features = []                               # List to hold features
    for ch_data in epoch:                       # Iterate over EEG channels in epoch
        freqs, psd = welch(ch_data, fs, nperseg=fs)  # Compute PSD via Welch's method
        for low, high in bands.values():        # For each frequency band
            idx = np.logical_and(freqs >= low, freqs <= high)  # Find freq indices within band
            features.append(np.mean(psd[idx]))  # Mean power in band appended as feature
    return features                             # Return feature vector

# === Load model and scaler ===
scaler = joblib.load('scaler.pkl')             # Load pre-fitted scaler
clf = joblib.load('classifier.pkl')            # Load trained classifier model

# === Setup BrainFlow board ===
params = BrainFlowInputParams()
params.serial_port = '/dev/cu.usbserial-DM03H3I9'  # Serial port for EEG board (update if needed)
board_id = BoardIds.CYTON_BOARD.value              # Board ID for Cyton EEG board
board = BoardShim(board_id, params)                # Initialize board shim

board.prepare_session()                            # Prepare board session
board.start_stream()                               # Start EEG data stream
print("🧠 EEG stream started...")

eeg_channels = BoardShim.get_eeg_channels(board_id)  # Get indices of EEG channels
buffer = np.empty((len(eeg_channels), 0))            # Initialize empty buffer for EEG data

# === Main Loop ===
try:
    while True:
        data = board.get_board_data()                 # Fetch latest data from board
        if data.shape[1] == 0:                        # If no new data available
            time.sleep(0.01)                          # Sleep briefly and retry
            continue

        eeg_data = data[eeg_channels, :]              # Extract EEG channel data
        buffer = np.hstack((buffer, eeg_data))        # Append new data to buffer horizontally

        while buffer.shape[1] >= window_size:         # While buffer has enough data for one window
            window = buffer[:, :window_size]          # Extract window from buffer
            filtered = bandpass_filter(window, 1, 50, sample_rate)  # Bandpass filter 1-50 Hz
            features = extract_bandpower_features(filtered, sample_rate)  # Extract bandpower features
            features_scaled = scaler.transform([features])  # Scale features
            prediction = clf.predict(features_scaled)       # Predict command
            command = prediction[0]

            print(f"Predicted command: {command}")          # Print prediction verbatim

            buffer = buffer[:, step_size:]                  # Remove processed data for next window

        time.sleep(0.01)                                    # Sleep briefly before next fetch

except KeyboardInterrupt:
    print("🛑 Stopping stream...")                          # Notify stopping
    board.stop_stream()                                     # Stop EEG data stream
    board.release_session()                                 # Release board session
    print("✅ Session closed.")                             # Confirm session closed
