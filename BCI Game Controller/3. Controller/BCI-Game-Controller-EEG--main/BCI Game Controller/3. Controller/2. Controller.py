import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, welch
from pynput.keyboard import Controller, Key

# === Settings ===
sample_rate = 250
window_size = int(sample_rate * 1.0)   # 1 second window
step_size = int(sample_rate * 0.5)     # 0.5 seconds step

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Label to command mapping ===
label_to_command = {
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
key_map = {
    'Move_Forward': 'w',
    'Move_Backward': 's',
    'Strafe_Left': 'a',
    'Strafe_Right': 'd',
    'Jump': Key.space,
    'Sneak': Key.shift,
    'Run': Key.ctrl_l,     # Left Ctrl key for Run (adjust if needed)
    'Turn_Left': Key.left,
    'Turn_Right': Key.right,
    'Stop': None           # No action for Stop
}

# === Filter & Feature Extraction ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def extract_bandpower_features(epoch, fs):
    features = []
    for ch_data in epoch:
        freqs, psd = welch(ch_data, fs, nperseg=fs)
        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.mean(psd[idx]))
    return features

# === Control Minecraft via keypress ===
keyboard = Controller()

def execute_command(command):
    key = key_map.get(command, None)
    if key is None:
        print(f"⏹️ No key mapped for command '{command}', skipping keypress.")
        return

    print(f"🕹️ Sending keypress for command: {command}")
    keyboard.press(key)
    time.sleep(0.5)  # Duration of key press (adjust as needed)
    keyboard.release(key)

# === Load model and scaler ===
scaler = joblib.load('scaler.pkl')
clf = joblib.load('classifier.pkl')

# === Setup BrainFlow board ===
params = BrainFlowInputParams()
params.serial_port = '/dev/cu.usbserial-DM03H3I9'  # Update this if needed
board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

board.prepare_session()
board.start_stream()
print("🧠 EEG stream started...")

eeg_channels = BoardShim.get_eeg_channels(board_id)
buffer = np.empty((len(eeg_channels), 0))

# === Main Loop ===
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
            pred_label = prediction[0]
            command = label_to_command.get(pred_label, None)

            if command is None:
                print(f"⚠️ Warning: No command mapped for label {pred_label}")
            else:
                print(f"🧠 Predicted command: {command} (Label: {pred_label})")
                execute_command(command)

            buffer = buffer[:, step_size:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("🛑 Stopping stream...")
    board.stop_stream()
    board.release_session()
    print("✅ Session closed.")
