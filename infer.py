"""
Infer: Active Alarm Detector LSTM

Loads the trained LSTM model and allows the user to test it with custom sequences.
"""

import numpy as np
from rich import box, print
from rich.console import Console
from rich.table import Table

from lstm import LstmNetwork, LstmParam

console = Console()

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

def main():
    model_path = "lstm_model.npz"
    try:
        param, _ = LstmParam.load(model_path)
    except FileNotFoundError:
        print(f"[bold red]Error: Model file '{model_path}' not found.[/bold red]")
        print("Please run `train.py` first to train and save the model.")
        return

    print(f"\n[bold green]Loaded model from {model_path}[/bold green]")

    network = LstmNetwork(param)

    print("\n[bold]Interactive Active Alarm Detector Inference[/bold]")
    print("Inputs: 0 (Normal), 1 (Error), 2 (Reset)")
    print("Enter a sequence of numbers separated by commas (or 'q' to quit).")
    print("Example: 0, 1, 0, 2, 0\n")

    while True:
        try:
            user_input = input("Enter sequence: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if not user_input:
                continue

            # Parse input
            parts = [p.strip() for p in user_input.split(',')]
            seq = [int(p) for p in parts if p]

            if not seq or not all(v in [0, 1, 2] for v in seq):
                print("[yellow]Invalid sequence. Please enter numbers 0, 1, or 2 separated by commas.[/yellow]")
                continue

            # Calculate actual state
            actual_states = active_alarm(seq)

            # Predict with LSTM
            network.reset_inputs()
            preds = []
            
            for val in seq:
                # Normalize input to [-1, 1] range
                norm_val = val - 1.0
                x = np.array([norm_val], dtype=float)
                network.add_input(x)
                
                # Get prediction and map back to [0, 1] space
                idx = len(network.inputs) - 1
                hidden_state = network.time_steps[idx].state.hidden_state[0]
                pred = (hidden_state + 1) / 2.0
                preds.append(pred)

            # Display results
            table = Table(
                title="Inference Results",
                box=box.SIMPLE_HEAD,
                show_lines=True,
            )
            table.add_column("Step", style="bold cyan", justify="center")
            table.add_column("Input Code", style="white")
            table.add_column("Actual Alarm", style="green")
            table.add_column("Predicted", style="yellow")
            table.add_column("Error", style="red")

            for t in range(len(seq)):
                table.add_row(
                    str(t + 1),
                    f"{seq[t]:d}",
                    "Active" if actual_states[t] == 1.0 else "Normal",
                    f"{preds[t]:.2f}",
                    f"{abs(preds[t] - actual_states[t]):.2f}"
                )

            console.print(table)
            print()

        except ValueError as e:
            print(f"[bold red]Error parsing input: {e}[/bold red]")
            print("Please ensure you enter only comma-separated numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
