"""
Train: Active Alarm Detector LSTM

Trains an LSTM model to act as a stateful switch.
Inputs:
  0: Normal Operation
  1: Error Triggered
  2: Reset Signal

Output:
  1 if the alarm is currently active, 0 otherwise.
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

def active_alarm(sequence):
    """Compute whether the alarm is active at each time step."""
    is_active = False
    result = []
    for v in sequence:
        if v == 1:
            is_active = True
        elif v == 2:
            is_active = False
        result.append(1.0 if is_active else 0.0)
    return result

def print_epoch_table(
    epoch, total_loss, sequences, all_targets_raw, network
):
    table = Table(
        title=f"Epoch {epoch:4d} | Total Loss: {total_loss:.5f}",
        box=box.SIMPLE_HEAD,
        show_lines=True,
    )
    table.add_column("Seq", style="bold cyan", justify="center")
    table.add_column("Input", style="white")
    table.add_column("Target Alarm", style="green")
    table.add_column("Predicted", style="yellow")
    table.add_column("Error", style="red")

    for i, (seq, targets_raw) in enumerate(zip(sequences, all_targets_raw)):
        # Normalize input to [-1, 1] range for LSTM
        # 0 -> -1
        # 1 -> 0
        # 2 -> 1
        inputs = [np.array([v - 1.0], dtype=float) for v in seq]
        
        network.reset_inputs()
        for x in inputs:
            network.add_input(x)

        # Output is expected to be [0, 1]. The LSTM outputs [-1, 1].
        # We can map the output from [-1, 1] back to [0, 1] space.
        preds = [
            (network.time_steps[t].state.hidden_state[0] + 1) / 2.0
            for t in range(len(seq))
        ]
        errors = [abs(p - t) for p, t in zip(preds, targets_raw)]

        if i < 2 or i >= len(sequences) - 2:
            table.add_row(
                str(i + 1),
                str(seq),
                str([f"{v:.0f}" for v in targets_raw]),
                str([f"{v:.2f}" for v in preds]),
                str([f"{v:.2f}" for v in errors]),
            )
        elif i == 2:
            table.add_row("...","...","...","...","...")

    console.print(table)


def main():
    np.random.seed(42)

    num_sequences = 100
    min_length = 5
    max_length = 15

    sequences = []
    
    # Add explicit edge cases
    sequences.extend([
        [0, 0, 0, 0],              # Never goes off
        [1, 0, 0, 0],              # Goes off immediately
        [1, 0, 2, 0],              # Goes off, then resets
        [0, 1, 0, 2, 0, 1],        # Multiple alarms
        [2, 2, 0, 1, 2],           # Reset before alarm
        [1, 1, 1, 2, 2],           # Duplicate alarms/resets
    ])

    for _ in range(num_sequences):
        length = np.random.randint(min_length, max_length + 1)
        # Weight probabilities: 60% normal, 20% error, 20% reset
        seq = np.random.choice([0, 1, 2], size=length, p=[0.6, 0.2, 0.2]).tolist()
        sequences.append(seq)

    all_targets_raw = [active_alarm(seq) for seq in sequences]

    # Map target outputs: target is [0, 1]. We map it to LSTM space which is [-1, 1]
    all_targets_norm = [
        [t * 2.0 - 1.0 for t in targets] for targets in all_targets_raw
    ]
    # Normalize inputs to [-1, 1] range
    all_inputs_norm = [
        [v - 1.0 for v in seq] for seq in sequences
    ]

    hidden_size = 16
    input_size = 1
    learning_rate = 0.05
    num_epochs = 1000

    print(f"\n[bold]Training for {num_epochs} epochs...[/bold]\n")

    param = LstmParam(hidden_size, input_size)
    network = LstmNetwork(param)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for seq_norm, targets_norm in zip(all_inputs_norm, all_targets_norm):
            inputs_array = [np.array([v], dtype=float) for v in seq_norm]
            network.reset_inputs()
            for x in inputs_array:
                network.add_input(x)

            total_loss += network.compute_loss_and_grads(targets_norm, MSELoss)
            param.apply_gradients(lr=learning_rate)

        if epoch == 0 or (epoch + 1) % 50 == 0:
            print_epoch_table(
                epoch + 1, total_loss, sequences, all_targets_raw, network
            )

    model_path = "lstm_model.npz"
    param.save(model_path)
    print(f"\n[bold green]Model saved to {model_path}[/bold green]")


if __name__ == "__main__":
    main()
