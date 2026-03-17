"""
Demo: Running Sum

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
from rich import box, print
from rich.console import Console
from rich.table import Table

from lstm import LstmNetwork, LstmParam

console = Console()


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


def print_sequences_table(sequences, all_targets_raw, all_targets_norm):
    table = Table(title="Training Sequences", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="bold cyan", justify="center")
    table.add_column("Input", style="white")
    table.add_column("Running Sum (target)", style="green")
    table.add_column("Normalized", style="dim")

    for i, (seq, raw, norm) in enumerate(
        zip(sequences, all_targets_raw, all_targets_norm)
    ):
        table.add_row(
            str(i + 1),
            str(seq),
            str([f"{v:+.2f}" for v in raw]),
            str([f"{v:+.2f}" for v in norm]),
        )

    console.print(table)


def print_epoch_table(
    epoch, total_loss, sequences, all_targets_raw, network, global_max
):
    table = Table(
        title=f"Epoch {epoch:4d} | Total Loss: {total_loss:.5f}",
        box=box.SIMPLE_HEAD,
        show_lines=True,
    )
    table.add_column("Seq", style="bold cyan", justify="center")
    table.add_column("Input", style="white")
    table.add_column("Target", style="green")
    table.add_column("Prediction", style="yellow")
    table.add_column("Error", style="red")

    for i, (seq, targets_raw) in enumerate(zip(sequences, all_targets_raw)):
        inputs = [np.array([v], dtype=float) for v in seq]
        network.reset_inputs()
        for x in inputs:
            network.add_input(x)

        preds = [
            network.time_steps[t].state.hidden_state[0] * global_max
            for t in range(len(seq))
        ]
        errors = [abs(p - t) for p, t in zip(preds, targets_raw)]

        table.add_row(
            str(i + 1),
            str(seq),
            str([f"{v:+.2f}" for v in targets_raw]),
            str([f"{v:+.2f}" for v in preds]),
            str([f"{v:.2f}" for v in errors]),
        )

    console.print(table)


def main():
    np.random.seed(42)

    # --- Training sequences ---
    sequences = [
        [1, -1, 1, -1, 1, -1],  # alternating: sum stays near 0
        [1, 1, 1, -1, -1, -1],  # rises then falls
        [1, 1, 1, 1, 1, 1],  # all positive: sum grows to 6
        [-1, -1, -1, 1, 1, 1],  # falls then rises
    ]

    # Build raw targets (running sums) for each sequence
    all_targets_raw = [running_sum(seq) for seq in sequences]

    # --- Normalize all targets into (-1, 1) ---
    global_max = max(abs(t) for targets in all_targets_raw for t in targets)
    all_targets_norm = [
        [t / global_max for t in targets] for targets in all_targets_raw
    ]

    print(
        f"\n[bold]Normalization scale:[/bold] ±{global_max} → divide all targets by {global_max}\n"
    )
    print_sequences_table(sequences, all_targets_raw, all_targets_norm)

    # --- Hyperparameters ---
    hidden_size = 32
    input_size = 1
    learning_rate = 0.05
    num_epochs = 500

    print(
        f"\n[bold]Training for {num_epochs} epochs on {len(sequences)} sequences...[/bold]\n"
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
            print_epoch_table(
                epoch + 1, total_loss, sequences, all_targets_raw, network, global_max
            )


if __name__ == "__main__":
    main()
