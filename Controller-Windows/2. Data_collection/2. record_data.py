import time
import numpy as np
import pandas as pd
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# === OpenBCI Setup ===
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = 'COMn'  # 🔧 Change this to your actual OpenBCI COM port
board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()
time.sleep(2)

eeg_channels = BoardShim.get_eeg_channels(board_id)
channel_names = BoardShim.get_eeg_names(board_id)

# === Trial Configuration ===
numbers = [str(i) for i in range(1, 11)]  # '1' to '10'
duration = 20             # recording time (seconds)
num_trials = 5            # trials per number
pause_message = ">> Press [Enter] to continue to the next trial..."

# ✅ Use safe Windows-compatible directory
data_dir = os.path.join(os.getcwd(), 'eeg_training_data')
os.makedirs(data_dir, exist_ok=True)

# === Main Trial Loop (Terminal-driven) ===
try:
    for i, number in enumerate(numbers):
        for rep in range(num_trials):
            print(f"\n🧠 Number {number} | Trial {rep+1}/{num_trials}")
            input(">> Get ready. When you're focused on the number, press [Enter] to start recording...")

            print(f">> THINKING: {number} — Recording for {duration} seconds")
            board.insert_marker(i + 1)
            board.get_board_data()  # clear buffer
            time.sleep(duration)

            data = board.get_board_data()
            eeg_data = data[eeg_channels, :]
            df = pd.DataFrame(eeg_data.T, columns=channel_names)
            df['label'] = number

            # Save path using
            filename = os.path.join(data_dir, f'trial_{rep+1}_number_{number}.csv')
            df.to_csv(filename, index=False)

            print(f"✅ Trial saved: {filename}")
            input(pause_message)

    print("\n🎉 Data collection complete! All files saved to:")
    print(os.path.abspath(data_dir))

finally:
    board.stop_stream()
    board.release_session()
