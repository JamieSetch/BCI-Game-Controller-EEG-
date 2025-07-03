import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, welch
from pynput.keyboard import Controller, Key

# === Settings ===
sample_rate = 250                                        # Sampling rate in Hz
window_size = int(sample_rate * 1.0)                     # 1 second window (250 samples)
step_size = int(sample_rate * 0.5)                       # 0.5 seconds step (125 samples)

bands = {                                                # Frequency bands for feature extraction
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Label to command mapping ===
label_to_command = {                                     # Map numeric labels to game commands
    1: 'Strafe_Right',
    2: 'Move_Backward',
    3: 'Strafe_Left',
    4: 'Move_Forward',
    5: 'Attack',
    6: 'Sneak',
    7: 'Jump',
    8: 'Turn_Left',
    9: 'Turn_Right',
    10: 'Stop'
}

# === Key mapping (including special keys) ===
key_map = {                                             # Map commands to keyboard keys
    'Move_Forward': 'w',
    'Move_Backward': 's',
    'Strafe_Left': 'a',
    'Strafe_Right': 'd',
    'Jump': Key.space,
    'Sneak': Key.shift,
    'Run': Key.ctrl_l,                                  # Left Ctrl key for Run (adjust if needed)
    'Turn_Left': Key.left,
    'Turn_Right': Key.right,
    'Stop': None                                        # No action for Stop command
}

# === Filter & Feature Extraction ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs                                      # Nyquist frequency
    b, a = butter(order, [low / nyq, high / nyq], btype='band')  # Filter coefficients
    return filtfilt(b, a, data, axis=1)                 # Apply zero-phase bandpass filter

def extract_bandpower_features(epoch, fs):
    features = []                                       # List to hold extracted features
    for ch_data in epoch:                               # For each EEG channel
        freqs, psd = welch(ch_data, fs, nperseg=fs)     # Compute power spectral density (PSD)
        for low, high in bands.values():                # For each frequency band
            idx = np.logical_and(freqs >= low, freqs <= high)  # Select PSD indices in band
            features.append(np.mean(psd[idx]))          # Append average power in band
    return features                                     # Return feature vector

# === Control Minecraft via keypress ===
keyboard = Controller()                                 # Initialize keyboard controller

def execute_command(command):
    key = key_map.get(command, None)                    # Get mapped key for command
    if key is None:                                     # If no key mapped
        print(f"⏹️ No key mapped for command '{command}', skipping keypress.")  # Notify skip
        return

    print(f"🕹️ Sending keypress for command: {command}")  # Indicate sending keypress
    keyboard.press(key)                                   # Press the key
    time.sleep(0.5)                                       # Hold key for 0.5 seconds (adjustable)
    keyboard.release(key)                                 # Release the key

# === Load model and scaler ===
scaler = joblib.load('scaler.pkl')                        # Load pre-fitted feature scaler
clf = joblib.load('classifier.pkl')                       # Load trained classifier

# === Setup BrainFlow board ===
params = BrainFlowInputParams()
params.serial_port = '/dev/cu.usbserial-DM03H3I9'        # Serial port (update if needed)
board_id = BoardIds.CYTON_BOARD.value                    # Board ID for Cyton board
board = BoardShim(board_id, params)                      # Initialize board shim

board.prepare_session()                                  # Prepare the EEG session
board.start_stream()                                     # Start data streaming
print("🧠 EEG stream started...")

eeg_channels = BoardShim.get_eeg_channels(board_id)      # Get EEG channel indices
buffer = np.empty((len(eeg_channels), 0))                # Initialize empty data buffer

# === Main Loop ===
try:
    while True:
        data = board.get_board_data()                     # Fetch new data from board
        if data.shape[1] == 0:                            # If no new samples
            time.sleep(0.01)                              # Wait briefly before retry
            continue

        eeg_data = data[eeg_channels, :]                  # Extract EEG channels from data
        buffer = np.hstack((buffer, eeg_data))            # Append new data to buffer

        while buffer.shape[1] >= window_size:             # While buffer has enough samples
            window = buffer[:, :window_size]              # Extract data window
            filtered = bandpass_filter(window, 1, 50, sample_rate)  # Bandpass filter 1-50 Hz
            features = extract_bandpower_features(filtered, sample_rate)  # Extract features
            features_scaled = scaler.transform([features])  # Scale features
            prediction = clf.predict(features_scaled)        # Predict label
            pred_label = prediction[0]
            command = label_to_command.get(pred_label, None)  # Map label to command

            if command is None:                             # If no command mapped
                print(f"⚠️ Warning: No command mapped for label {pred_label}")  # Warn
            else:
                print(f"🧠 Predicted command: {command} (Label: {pred_label})")  # Print prediction
                execute_command(command)                     # Execute mapped keypress

            buffer = buffer[:, step_size:]                   # Remove processed samples

        time.sleep(0.01)                                     # Sleep briefly to reduce CPU load

except KeyboardInterrupt:
    print("🛑 Stopping stream...")                           # Notify stopping
    board.stop_stream()                                      # Stop EEG data stream
    board.release_session()                                  # Release board session
    print("✅ Session closed.")                              # Confirm session closure
