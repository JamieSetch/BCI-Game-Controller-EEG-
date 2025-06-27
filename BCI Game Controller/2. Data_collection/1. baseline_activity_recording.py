from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import numpy as np
import pandas as pd
import os

# === Setup OpenBCI Board Connection ===
BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.serial_port =  # Add your OpenBCI serial port
board_id = BoardIds.CYTON_BOARD.value

board = BoardShim(board_id, params)
board.prepare_session()

# === Output Directory ===
data_dir = 'eeg_training_data'
os.makedirs(data_dir, exist_ok=True)

# === Start Stream ===
board.start_stream()
time.sleep(2)

eeg_channels = BoardShim.get_eeg_channels(board_id)
channel_names = BoardShim.get_eeg_names(board_id)

# === Baseline Instructions ===
print("\n🧘 Starting 4-minute baseline EEG recording with eyes OPEN (relax and remain still)...")
time.sleep(240)  # 4 minutes eyes open baseline

# === Stop and Collect Baseline Data ===
data = board.get_board_data()
board.stop_stream()
board.release_session()

eeg_data = data[eeg_channels, :]
df = pd.DataFrame(eeg_data.T, columns=channel_names)
df['label'] = 'Baseline_EyesOpen'

baseline_file = f'{data_dir}/baseline_eyes_open_recording.csv'
df.to_csv(baseline_file, index=False)

print("\n✅ Baseline recording complete. File saved to:", os.path.abspath(baseline_file))
