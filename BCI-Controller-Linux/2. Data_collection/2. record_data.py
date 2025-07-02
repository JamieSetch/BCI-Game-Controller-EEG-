import time
import numpy as np
import pandas as pd
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# === OpenBCI Setup ===
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()

# === Set the correct serial port for your system ===
# Linux: e.g., '/dev/ttyUSB0' | Windows: 'COM3', etc.
params.serial_port = '/dev/ttyUSB0'  # <-- CHANGE THIS for your system

if not params.serial_port:
    raise ValueError("❌ You must set the serial port for your OpenBCI device (e.g., '/dev/ttyUSB0' or 'COM3').")

board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

board.prepare_session()
board.start_stream()
time.sleep(2)  # Give the stream time to stabilize

eeg_channels = BoardShim.get_eeg_channels(board_id)
channel_names = BoardShim.get_eeg_names(board_id)

# === Trial Configuration ===
numbers = [str(i) for i in range(1, 11)]  # '1' to '10'
duration = 20             # seconds per trial
num_trials = 5            # number of trials per digit
pause_message = ">> Press [Enter] to continue to the next trial..."

data_dir = 'eeg_training_data'
os.makedirs(data_dir, exist_ok=True)

print("📋 EEG Number Thinking Task Started")
print("👉 Focus on the number shown when prompted.")
print("Each trial will record 20 seconds of EEG.\n")

try:
    for i, number in enumerate(numbers):
        for rep in range(num_trials):
            print(f"\n🧠 Number {number} | Trial {rep+1}/{num_trials}")
            input(">> Get ready. When you're focused on the number, press [Enter] to start recording...")

            print(f"🎯 THINKING: {number} — Recording for {duration} seconds...")
            board.insert_marker(i + 1)  # Add a marker for the label (optional)
            board.get_board_data()  # Clear buffer
            time.sleep(duration)

            data = board.get_board_data()
            eeg_data = data[eeg_channels, :]
            df = pd.DataFrame(eeg_data.T, columns=channel_names)
            df['label'] = number

            filename = f'{data_dir}/trial_{rep+1}_number_{number}.csv'
            df.to_csv(filename, index=False)

            print(f"✅ Trial saved: {filename}")
            input(pause_message)

    print("\n🎉 Data collection complete! All files saved to:")
    print(os.path.abspath(data_dir))

finally:
    print("\n🛑 Cleaning up BrainFlow session...")
    board.stop_stream()
    board.release_session()
    print("✅ Session ended cleanly.")
