"""
Demo: Running Sum (Multi-Sequence)

Goal: Train the LSTM to compute the running sum of sequences.

At each time step, the LSTM receives a value and must output
the cumulative sum of all values seen so far.

Example:
  Input:       1   1   1  -1  -1  -1
  Running sum: 1   2   3   2   1   0

Since the LSTM's output is bounded to (-1, +1), we normalize the
targets before training and denormalize predictions for display.
"""

import numpy as np
from rich import print
from lstm import LstmParam, LstmNetwork


class MSELoss:
    @classmethod
    def loss(cls, pred, target):
        return (pred[0] - target) ** 2

    @classmethod
    def bottom_diff(cls, pred, target):
        grad = np.zeros_like(pred)
        grad[0] = 2 * (pred[0] - target)
        return grad


def running_sum(sequence):
    """Compute the cumulative sum at each time step."""
    total = 0.0
    result = []
    for v in sequence:
        total += v
        result.append(total)
    return result


def main():
    np.random.seed(42)

    # --- Training sequences ---
    sequences = [
        [1, -1, 1, -1, 1, -1],  # alternating → sum stays near 0
        [1, 1, 1, -1, -1, -1],  # rises then falls
        [1, 1, 1, 1, 1, 1],  # all positive → sum grows to 6
        [-1, -1, -1, 1, 1, 1],  # falls then rises
    ]

    # Build raw targets (running sums) for each sequence
    all_targets_raw = [running_sum(seq) for seq in sequences]

    # --- Normalize all targets into (-1, 1) ---
    # Find the largest absolute value across ALL sequences
    global_max = max(abs(t) for targets in all_targets_raw for t in targets)
    all_targets_norm = [
        [t / global_max for t in targets] for targets in all_targets_raw
    ]

    print(f"Normalization scale: {global_max} (divide targets by this)")
    for i, (seq, raw, norm) in enumerate(
        zip(sequences, all_targets_raw, all_targets_norm)
    ):
        print(f"\nSequence {i + 1}: {seq}")
        print(f"  Running sum:       {raw}")
        print(f"  Normalized target: {[round(n, 2) for n in norm]}")

    # --- Hyperparameters ---
    hidden_size = 32
    input_size = 1
    learning_rate = 0.05
    num_epochs = 500

    print(
        f"\n\nTraining for {num_epochs} epochs on all {len(sequences)} sequences...\n"
    )

    # --- Initialize LSTM ---
    param = LstmParam(hidden_size, input_size)
    network = LstmNetwork(param)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        total_loss = 0.0

        for seq, targets_norm in zip(sequences, all_targets_norm):
            inputs = [np.array([v], dtype=float) for v in seq]

            network.reset_inputs()
            for x in inputs:
                network.add_input(x)

            total_loss += network.compute_loss_and_grads(targets_norm, MSELoss)
            param.apply_gradients(lr=learning_rate)

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:4d} | Total Loss: {total_loss:.5f}")
            for i, (seq, targets_raw, targets_norm) in enumerate(
                zip(sequences, all_targets_raw, all_targets_norm)
            ):
                inputs = [np.array([v], dtype=float) for v in seq]
                network.reset_inputs()
                for x in inputs:
                    network.add_input(x)

                # Denormalize predictions back to real scale
                preds = [
                    network.time_steps[t].state.hidden_state[0] * global_max
                    for t in range(len(seq))
                ]
                print(f"  Seq {i + 1} {seq}")
                print(f"    Target: {[f'{v:+.2f}' for v in targets_raw]}")
                print(f"    Pred:   {[f'{v:+.2f}' for v in preds]}")
            print()


if __name__ == "__main__":
    main()
