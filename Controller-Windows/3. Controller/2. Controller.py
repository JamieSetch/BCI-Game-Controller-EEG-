import time
import numpy as np
import joblib
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, welch
from pynput.keyboard import Controller, Key

# === Configuration ===
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

LABEL_TO_COMMAND = {
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

KEY_MAP = {
    'Move_Forward': 'w',
    'Move_Backward': 's',
    'Strafe_Left': 'a',
    'Strafe_Right': 'd',
    'Jump': Key.space,
    'Sneak': Key.shift,
    'Run': Key.ctrl_l,      # Optional: add if needed
    'Turn_Left': Key.left,
    'Turn_Right': Key.right,
    'Attack': Key.ctrl_l,
    'Stop': None            # No key press for Stop
}

# === Helper Functions ===
def bandpass_filter(data, low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def extract_bandpower_features(epoch, fs):
    features = []
    for ch_data in epoch:
        freqs, psd = welch(ch_data, fs, nperseg=fs)
        for low, high in BANDS.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.mean(psd[idx]))
    return features

keyboard = Controller()

def execute_command(command):
    key = KEY_MAP.get(command, None)
    if key is None:
        print(f"⏹️ Command '{command}' has no key binding.")
        return
    print(f"🕹️ Executing: {command}")
    keyboard.press(key)
    time.sleep(0.5)
    keyboard.release(key)

# === Load Classifier and Scaler ===
try:
    scaler = joblib.load('scaler.pkl')
    clf = joblib.load('classifier.pkl')
    print("✅ Model and scaler loaded.")
except Exception as e:
    raise RuntimeError(f"❌ Could not load model/scaler: {e}")

# === Initialize BrainFlow ===
params = BrainFlowInputParams()
params.serial_port = 'COM3'  # ⚠️ Replace with your actual COM port
board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

try:
    BoardShim.enable_dev_board_logger()
    board.prepare_session()
    board.start_stream()
    print("🧠 EEG stream started...")

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    buffer = np.empty((len(eeg_channels), 0))

    # === Main Loop ===
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
            pred_label = prediction[0]
            command = LABEL_TO_COMMAND.get(pred_label)

            if command:
                print(f"🧠 Predicted command: {command} (Label: {pred_label})")
                execute_command(command)
            else:
                print(f"⚠️ Unknown label: {pred_label}")

            buffer = buffer[:, STEP_SIZE:]

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🛑 Stopping stream...")

finally:
    board.stop_stream()
    board.release_session()
    print("✅ EEG session ended.")
