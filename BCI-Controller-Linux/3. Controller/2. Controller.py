import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, welch
from pynput.keyboard import Controller, Key

# === Sampling Settings ===
sample_rate = 250
window_size = int(sample_rate * 1.0)   # 1 second
step_size = int(sample_rate * 0.5)     # 0.5 seconds

# === EEG Frequency Bands ===
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# === Label to Command Mapping ===
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

# === Command to Keyboard Key Mapping ===
key_map = {
    'Move_Forward': 'w',
    'Move_Backward': 's',
    'Strafe_Left': 'a',
    'Strafe_Right': 'd',
    'Jump': Key.space,
    'Sneak': Key.shift,
    'Attack': Key.enter,
    'Turn_Left': Key.left,
    'Turn_Right': Key.right,
    'Stop': None  # Do nothing
}

# === Filtering Function ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

# === Feature Extraction ===
def extract_bandpower_features(epoch, fs):
    features = []
    for ch_data in epoch:
        freqs, psd = welch(ch_data, fs, nperseg=fs)
        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.mean(psd[idx]))
    return features

# === Keyboard Controller ===
keyboard = Controller()

def execute_command(command):
    key = key_map.get(command)
    if key is None:
        print(f"⏹️ No action mapped for '{command}'.")
        return

    try:
        print(f"🕹️ Executing: {command}")
        keyboard.press(key)
        time.sleep(0.3)  # Adjust press duration as needed
        keyboard.release(key)
    except Exception as e:
        print(f"⚠️ Key press error: {e}")

# === Load Pretrained Classifier and Scaler ===
scaler = joblib.load('scaler.pkl')
clf = joblib.load('classifier.pkl')

# === Setup BrainFlow Board ===
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'  # <-- Update to match your Cyton port (check with `ls /dev/ttyUSB*`)

board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

BoardShim.enable_dev_board_logger()

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
            command = label_to_command.get(pred_label)

            if command:
                print(f"🧠 Predicted: {command} (Label: {pred_label})")
                execute_command(command)
            else:
                print(f"❓ Unknown label: {pred_label}")

            buffer = buffer[:, step_size:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🛑 Keyboard interrupt detected. Cleaning up...")
    board.stop_stream()
    board.release_session()
    print("✅ Session ended cleanly.")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    board.stop_stream()
    board.release_session()
