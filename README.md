# LSTM from Scratch

A minimal, educational implementation of a Long Short-Term Memory (LSTM) network from scratch using NumPy. This repository is adapted from [nicodjimenez/lstm](https://github.com/nicodjimenez/lstm) and has been refactored with more descriptive variable names and a modular structure to aid understandability.

## Key Refactorings

- **Improved Readability**: Replaced abbreviated names (e.g., `wi`, `wf`, `wo`, `wc`) with descriptive ones (`weight_input_gate`, `weight_forget_gate`, `weight_output_gate`, `weight_cell`).
- **Modular Architecture**: Separated the core LSTM logic into `lstm.py` and provided specialized scripts for training (`train.py`) and inference (`infer.py`).
- **Stateful Applications**: Includes a concrete example of a stateful "Active Alarm Detector".

## Project Structure

- **`lstm.py`**: The core LSTM logic, including parameter management, state tracking, and Backpropagation Through Time (BPTT).
- **`train.py`**: Training script for the "Active Alarm Detector" task.
- **`infer.py`**: Interactive inference script for testing the trained model.

## How it Works: Active Alarm Detector

The provided example trains the LSTM to act as a stateful switch:
- **Input 1 (Error)**: Triggers the alarm (Output -> 1.0).
- **Input 2 (Reset)**: Disables the alarm (Output -> 0.0).
- **Input 0 (Normal)**: Maintains the current state.

The LSTM learns to "remember" the alarm state across multiple time steps without explicit state management in the inference logic.

## Usage

1. **Install Dependencies**:

   ```bash
   pip install numpy rich
   ```

2. **Train the Model**:

   ```bash
   python train.py
   ```

   This will train the model and save the weights to `lstm_model.npz`.

3. **Run Inference**:

   ```bash
   python infer.py
   ```

   Enter comma-separated sequences (e.g., `0, 1, 0, 2, 0`) to test the model's memory.

## Credits

Original implementation by [Nico Jimenez](https://github.com/nicodjimenez/lstm).
For a deep dive into the math behind the backpropagation, see the [original blog post](http://nicodjimenez.github.io/2014/08/08/lstm.html).
