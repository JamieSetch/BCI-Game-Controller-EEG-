import time                                                      
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds  

BoardShim.enable_dev_board_logger()                             

params = BrainFlowInputParams()                                  # Create an object to hold board connection parameters
params.serial_port =                                             # Specify the serial port of your OpenBCI dongle (e.g., 'COM3' or '/dev/ttyUSB0')

board = BoardShim(BoardIds.CYTON_BOARD.value, params)            # Initialize the board object for Cyton with the given parameters
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)  # Get indices of EEG channels for the Cyton board
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)  # Get the board's sampling rate (usually 250 Hz)

board.prepare_session()                                          # Prepare the board session (initialize and ready hardware)
board.start_stream()                                             # Start streaming data from the board
print("Streaming started... (Ctrl+C to stop)")                   # Notify user that streaming has started

try:
    while True:                                                  # Infinite loop to continuously read and print EEG data
        time.sleep(0.5)                                          # Wait for 0.5 seconds to accumulate new data
        data = board.get_current_board_data(100)                 # Get the latest 100 samples (~0.4s of data)

        print("Latest EEG values (last sample per channel):")    # Print header for data output
        for i, ch in enumerate(eeg_channels):                    # Loop through each EEG channel
            if data.shape[1] > 0:                                # If there is data available
                print(f"Channel {i+1} (index {ch}): {data[ch][-1]:.2f} µV")  # Print last value for this channel
            else:                                                # If no data received
                print(f"Channel {i+1} (index {ch}): No data")    # Inform no data for this channel
        print("-" * 40)                                          # Print separator for readability
except KeyboardInterrupt:                                        # Handle Ctrl+C to safely stop streaming
    print("Interrupted by user.")                                # Notify user of interruption
finally:
    board.stop_stream()                                          # Stop the data stream
    board.release_session()                                      # Release resources and end session
    print("Session ended.")                                      # Notify user that session has ended
