from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import numpy as np
import pandas as pd
import os
import sys

# === Setup OpenBCI Board Connection ===
BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()

# === Set the correct serial port for your system ===
# Example for Linux: /dev/ttyUSB0 or /dev/ttyACM0
params.serial_port = '/dev/ttyUSB0'  # <-- CHANGE THIS to match your system

if not params.serial_port:
    print("❌ Error: Serial port not specified. Set 'params.serial_port' to your OpenBCI device (e.g., '/dev/ttyUSB0').")
    sys.exit(1)

board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

try:
    board.prepare_session()

    # === Output Directory ===
    data_dir = 'eeg_training_data'
    os.makedirs(data_dir, exist_ok=True)

    # === Start Stream ===
    board.start_stream()
    time.sleep(2)  # small buffer to allow stream to stabilize

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    channel_names = BoardShim.get_eeg_names(board_id)

    # === Baseline Instructions ===
    print("\n🧘 Starting 4-minute baseline EEG recording with eyes OPEN...")
    print("👉 Please sit still and relax with your eyes open.")
    time.sleep(240)  # 4 minutes of recording

    # === Stop and Collect Data ===
    print("📥 Collecting data and saving...")
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_data = data[eeg_channels, :]
    df = pd.DataFrame(eeg_data.T, columns=channel_names)
    df['label'] = 'Baseline_EyesOpen'

    baseline_file = os.path.join(data_dir, 'baseline_eyes_open_recording.csv')
    df.to_csv(baseline_file, index=False)

    print("\n✅ Baseline recording complete.")
    print("📁 File saved to:", os.path.abspath(baseline_file))

except Exception as e:
    print("❌ Error during recording:", str(e))
    board.release_session()
    sys.exit(1)
