from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import numpy as np
import pandas as pd
import os

# === Setup OpenBCI Board Connection ===
BoardShim.enable_dev_board_logger()                                       # Enable logging for development/debugging

params = BrainFlowInputParams()                                           # Create parameter object to hold connection info
params.serial_port =                                                     # Add your OpenBCI serial port here (e.g., 'COM3' or '/dev/ttyUSB0')
board_id = BoardIds.CYTON_BOARD.value                                     # Set the board ID to use the Cyton board

board = BoardShim(board_id, params)                                       # Initialize the board connection with parameters
board.prepare_session()                                                  # Prepare the session (setup buffers and connection)

# === Output Directory ===
data_dir = 'eeg_training_data'                                            # Set directory name for saving EEG data
os.makedirs(data_dir, exist_ok=True)                                      # Create the directory if it doesn't exist

# === Start Stream ===
board.start_stream()                                                     # Start streaming EEG data from the board
time.sleep(2)                                                            # Allow some time to stabilize the stream

eeg_channels = BoardShim.get_eeg_channels(board_id)                      # Get the EEG channel indices for the board
channel_names = BoardShim.get_eeg_names(board_id)                        # Get human-readable names for the EEG channels

# === Baseline Instructions ===
print("\n🧘 Starting 4-minute baseline EEG recording with eyes OPEN (relax and remain still)...")  # Notify user of start
time.sleep(240)                                                          # Record baseline for 4 minutes (240 seconds)

# === Stop and Collect Baseline Data ===
data = board.get_board_data()                                            # Retrieve all data collected during the stream
board.stop_stream()                                                     # Stop the data stream
board.release_session()                                                 # Release resources and end session

eeg_data = data[eeg_channels, :]                                         # Extract only EEG channel data from full array
df = pd.DataFrame(eeg_data.T, columns=channel_names)                    # Convert EEG data to a DataFrame with column names
df['label'] = 'Baseline_EyesOpen'                                       # Add a label column for supervised learning

baseline_file = f'{data_dir}/baseline_eyes_open_recording.csv'          # Define the output file path
df.to_csv(baseline_file, index=False)                                   # Save the DataFrame to a CSV file

print("\n✅ Baseline recording complete. File saved to:", os.path.abspath(baseline_file))  # Confirm save location to user
