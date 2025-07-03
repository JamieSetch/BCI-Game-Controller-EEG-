import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# === Set up BrainFlow ===
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()

# ⚠️ Set the correct Linux serial port for your OpenBCI dongle
# You can find it using: ls /dev/ttyUSB* or dmesg | grep tty
params.serial_port = '/dev/ttyUSB0'  # <- change if needed (e.g., '/dev/ttyUSB1')

# === Initialize board ===
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)

board.prepare_session()
board.start_stream()
print("📡 EEG Streaming started... (Ctrl+C to stop)")

try:
    while True:
        time.sleep(0.5)  # wait to accumulate data
        data = board.get_current_board_data(100)  # ~0.4s of data
        print("🧠 Latest EEG values (last sample per channel):")
        for i, ch in enumerate(eeg_channels):
            if data.shape[1] > 0:
                print(f"Channel {i+1} (index {ch}): {data[ch][-1]:.2f} µV")
            else:
                print(f"Channel {i+1} (index {ch}): No data")
        print("-" * 40)
except KeyboardInterrupt:
    print("🛑 Interrupted by user.")
finally:
    board.stop_stream()
    board.release_session()
    print("✅ Session ended.")
