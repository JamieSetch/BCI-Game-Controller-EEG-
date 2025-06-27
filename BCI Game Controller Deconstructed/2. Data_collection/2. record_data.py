import time
import numpy as np
import pandas as pd
import os
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# === OpenBCI Setup ===
BoardShim.enable_dev_board_logger()                                        # Enable logging for BrainFlow (debugging)
params = BrainFlowInputParams()                                            # Create an object to hold board connection parameters
params.serial_port =                                                       # Add your OpenBCI serial port here (e.g., 'COM3' or '/dev/ttyUSB0')
board_id = BoardIds.CYTON_BOARD.value                                      # Use the CYTON board ID from BrainFlow
board = BoardShim(board_id, params)                                        # Initialize the board connection
board.prepare_session()                                                    # Prepare the session (allocate buffers, etc.)
board.start_stream()                                                       # Begin streaming EEG data
time.sleep(2)                                                              # Wait for stream to stabilize

eeg_channels = BoardShim.get_eeg_channels(board_id)                        # Get indices of EEG channels
channel_names = BoardShim.get_eeg_names(board_id)                          # Get human-readable names of EEG channels

# === Trial Configuration ===
numbers = [str(i) for i in range(1, 11)]                                   # Create a list of numbers as strings from '1' to '10'
duration = 20                                                              # Duration of each trial in seconds
num_trials = 5                                                             # Number of repetitions per number
pause_message = ">> Press [Enter] to continue to the next trial..."        # Message to show after each trial

data_dir = 'eeg_training_data'                                             # Directory to save EEG trial data
os.makedirs(data_dir, exist_ok=True)                                       # Create the directory if it doesn't already exist

# === Main Trial Loop (Terminal-driven) ===
try:
    for i, number in enumerate(numbers):                                   # Loop through each number (e.g., '1' to '10')
        for rep in range(num_trials):                                      # Loop through each repetition for the current number
            print(f"\n🧠 Number {number} | Trial {rep+1}/{num_trials}")    # Display current trial information
            input(">> Get ready. When you're focused on the number, press [Enter] to start recording...")  # Prompt user to start

            print(f">> THINKING: {number} — Recording for {duration} seconds")  # Notify recording start
            board.insert_marker(i + 1)                                      # Insert marker to mark the start of this trial
            board.get_board_data()                                          # Clear existing data buffer
            time.sleep(duration)                                            # Wait for the trial duration while recording

            data = board.get_board_data()                                   # Get the recorded EEG data from the buffer
            eeg_data = data[eeg_channels, :]                                # Extract only EEG channel data
            df = pd.DataFrame(eeg_data.T, columns=channel_names)            # Convert to DataFrame with proper column names
            df['label'] = number                                            # Add a label column for supervised learning

            filename = f'{data_dir}/trial_{rep+1}_number_{number}.csv'      # Construct filename for saving the trial
            df.to_csv(filename, index=False)                                # Save DataFrame to CSV file

            print(f"✅ Trial saved: {filename}")                            # Confirm file save
            input(pause_message)                                            # Wait for user to proceed to next trial

    print("\n🎉 Data collection complete! All files saved to:")             # Notify user of completion
    print(os.path.abspath(data_dir))                                        # Print absolute path to saved data

finally:
    board.stop_stream()                                                     # Stop EEG data stream
    board.release_session()                                                 # Release board resources and end session
