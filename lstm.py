import numpy as np


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def sigmoid_grad(values):
    return values * (1 - values)


def tanh_grad(values):
    return 1.0 - values**2


def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))


class LstmParam:
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        concat_size = input_size + hidden_size

        # Weight matrices for each gate/node (shape: hidden_size x concat_size)
        self.weight_cell = xavier_init(concat_size, hidden_size)
        self.weight_input_gate = xavier_init(concat_size, hidden_size)
        self.weight_forget_gate = xavier_init(concat_size, hidden_size)
        self.weight_output_gate = xavier_init(concat_size, hidden_size)

        # Bias vectors for each gate/node
        self.bias_cell = np.zeros(hidden_size)
        self.bias_input_gate = np.ones(hidden_size)
        self.bias_forget_gate = np.zeros(hidden_size)
        self.bias_output_gate = np.zeros(hidden_size)

        # Gradient accumulators (derivative of loss w.r.t. each parameter)
        self.weight_cell_grad = np.zeros((hidden_size, concat_size))
        self.weight_input_gate_grad = np.zeros((hidden_size, concat_size))
        self.weight_forget_gate_grad = np.zeros((hidden_size, concat_size))
        self.weight_output_gate_grad = np.zeros((hidden_size, concat_size))
        self.bias_cell_grad = np.zeros(hidden_size)
        self.bias_input_gate_grad = np.zeros(hidden_size)
        self.bias_forget_gate_grad = np.zeros(hidden_size)
        self.bias_output_gate_grad = np.zeros(hidden_size)

    def apply_gradients(self, lr=1):
        self.weight_cell -= lr * self.weight_cell_grad
        self.weight_input_gate -= lr * self.weight_input_gate_grad
        self.weight_forget_gate -= lr * self.weight_forget_gate_grad
        self.weight_output_gate -= lr * self.weight_output_gate_grad
        self.bias_cell -= lr * self.bias_cell_grad
        self.bias_input_gate -= lr * self.bias_input_gate_grad
        self.bias_forget_gate -= lr * self.bias_forget_gate_grad
        self.bias_output_gate -= lr * self.bias_output_gate_grad

        # Reset gradient accumulators to zero after each update
        self.weight_cell_grad = np.zeros_like(self.weight_cell)
        self.weight_input_gate_grad = np.zeros_like(self.weight_input_gate)
        self.weight_forget_gate_grad = np.zeros_like(self.weight_forget_gate)
        self.weight_output_gate_grad = np.zeros_like(self.weight_output_gate)
        self.bias_cell_grad = np.zeros_like(self.bias_cell)
        self.bias_input_gate_grad = np.zeros_like(self.bias_input_gate)
        self.bias_forget_gate_grad = np.zeros_like(self.bias_forget_gate)
        self.bias_output_gate_grad = np.zeros_like(self.bias_output_gate)


class LstmState:
    def __init__(self, hidden_size, input_size):
        self.cell_input = np.zeros(hidden_size)  # g: new candidate info (tanh)
        self.input_gate = np.zeros(hidden_size)  # i: how much new info to write
        self.forget_gate = np.zeros(hidden_size)  # f: how much old memory to keep
        self.output_gate = np.zeros(hidden_size)  # o: what to expose from memory
        self.cell_state = np.zeros(
            hidden_size
        )  # s: the long-term memory ("conveyor belt")
        self.hidden_state = np.zeros(hidden_size)  # h: the output at this time step
        self.grad_prev_hidden = np.zeros_like(
            self.hidden_state
        )  # gradient flowing back to h(t-1)
        self.grad_prev_cell = np.zeros_like(
            self.cell_state
        )  # gradient flowing back to s(t-1)


class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        self.state = lstm_state
        self.param = lstm_param
        self.concat_input = None  # x(t) concatenated with h(t-1), saved for backprop

    def forward(self, x, prev_cell_state=None, prev_hidden_state=None):
        # Use zero vectors for the first time step (no previous state)
        if prev_cell_state is None:
            prev_cell_state = np.zeros_like(self.state.cell_state)
        if prev_hidden_state is None:
            prev_hidden_state = np.zeros_like(self.state.hidden_state)

        # Save previous states for use in backprop
        self.prev_cell_state = prev_cell_state
        self.prev_hidden_state = prev_hidden_state

        # Concatenate current input x(t) with previous hidden state h(t-1)
        concat_input = np.hstack((x, prev_hidden_state))

        # Compute gate activations
        self.state.cell_input = np.tanh(
            np.dot(self.param.weight_cell, concat_input) + self.param.bias_cell
        )
        self.state.input_gate = sigmoid(
            np.dot(self.param.weight_input_gate, concat_input)
            + self.param.bias_input_gate
        )
        self.state.forget_gate = sigmoid(
            np.dot(self.param.weight_forget_gate, concat_input)
            + self.param.bias_forget_gate
        )
        self.state.output_gate = sigmoid(
            np.dot(self.param.weight_output_gate, concat_input)
            + self.param.bias_output_gate
        )

        # Update cell state: keep old memory (forget gate) + write new info (input gate)
        self.state.cell_state = (
            self.state.cell_input * self.state.input_gate
            + prev_cell_state * self.state.forget_gate
        )
        # Compute hidden state: expose filtered memory through output gate
        self.state.hidden_state = (
            np.tanh(self.state.cell_state) * self.state.output_gate
        )

        self.concat_input = concat_input

    def backward(self, grad_hidden, grad_cell):
        # grad_hidden: gradient of loss w.r.t. hidden_state at this time step
        # grad_cell:   gradient flowing back along the cell state (constant error carousel)

        cell_state_tanh = np.tanh(self.state.cell_state)

        # Backprop through cell state tanh and output gate
        delta_cell_state = (
            self.state.output_gate * grad_hidden * (1 - cell_state_tanh**2) + grad_cell
        )
        delta_output_gate = cell_state_tanh * grad_hidden

        # Backprop into each gate's pre-activation value
        delta_input_gate = self.state.cell_input * delta_cell_state
        delta_cell_input = self.state.input_gate * delta_cell_state
        delta_forget_gate = self.prev_cell_state * delta_cell_state

        # Apply activation function derivatives (chain rule)
        delta_input_gate_raw = sigmoid_grad(self.state.input_gate) * delta_input_gate
        delta_forget_gate_raw = sigmoid_grad(self.state.forget_gate) * delta_forget_gate
        delta_output_gate_raw = sigmoid_grad(self.state.output_gate) * delta_output_gate
        delta_cell_input_raw = tanh_grad(self.state.cell_input) * delta_cell_input

        # Accumulate weight gradients (outer product: delta x concat_input^T)
        self.param.weight_input_gate_grad += np.outer(
            delta_input_gate_raw, self.concat_input
        )
        self.param.weight_forget_gate_grad += np.outer(
            delta_forget_gate_raw, self.concat_input
        )
        self.param.weight_output_gate_grad += np.outer(
            delta_output_gate_raw, self.concat_input
        )
        self.param.weight_cell_grad += np.outer(delta_cell_input_raw, self.concat_input)

        # Accumulate bias gradients
        self.param.bias_input_gate_grad += delta_input_gate_raw
        self.param.bias_forget_gate_grad += delta_forget_gate_raw
        self.param.bias_output_gate_grad += delta_output_gate_raw
        self.param.bias_cell_grad += delta_cell_input_raw

        # Compute gradient w.r.t. the concatenated input to propagate backwards
        grad_concat_input = np.zeros_like(self.concat_input)
        grad_concat_input += np.dot(
            self.param.weight_input_gate.T, delta_input_gate_raw
        )
        grad_concat_input += np.dot(
            self.param.weight_forget_gate.T, delta_forget_gate_raw
        )
        grad_concat_input += np.dot(
            self.param.weight_output_gate.T, delta_output_gate_raw
        )
        grad_concat_input += np.dot(self.param.weight_cell.T, delta_cell_input_raw)

        # Split concat gradient: the h(t-1) portion flows back to the previous node
        self.state.grad_prev_cell = delta_cell_state * self.state.forget_gate
        self.state.grad_prev_hidden = grad_concat_input[self.param.input_size :]


class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.time_steps = []  # one LstmNode per time step
        self.inputs = []  # input sequence

    def compute_loss_and_grads(self, target_list, loss_layer):
        """
        Run backpropagation through time (BPTT).
        Computes gradients for all time steps given the targets.
        Does NOT update parameters — call lstm_param.apply_gradients() separately.
        """
        assert len(target_list) == len(self.inputs)

        # Start from the last time step
        idx = len(self.inputs) - 1
        loss = loss_layer.loss(
            self.time_steps[idx].state.hidden_state, target_list[idx]
        )
        grad_hidden = loss_layer.bottom_diff(
            self.time_steps[idx].state.hidden_state, target_list[idx]
        )
        # Cell state has no direct gradient from loss at the final step
        grad_cell = np.zeros(self.lstm_param.hidden_size)
        self.time_steps[idx].backward(grad_hidden, grad_cell)
        idx -= 1

        # Walk backwards through remaining time steps
        while idx >= 0:
            loss += loss_layer.loss(
                self.time_steps[idx].state.hidden_state, target_list[idx]
            )
            # Gradient from the loss at this time step + gradient from the next time step
            grad_hidden = loss_layer.bottom_diff(
                self.time_steps[idx].state.hidden_state, target_list[idx]
            )
            grad_hidden += self.time_steps[idx + 1].state.grad_prev_hidden
            grad_cell = self.time_steps[idx + 1].state.grad_prev_cell
            self.time_steps[idx].backward(grad_hidden, grad_cell)
            idx -= 1

        return loss

    def reset_inputs(self):
        self.inputs = []

    def add_input(self, x):
        self.inputs.append(x)
        if len(self.inputs) > len(self.time_steps):
            lstm_state = LstmState(
                self.lstm_param.hidden_size, self.lstm_param.input_size
            )
            self.time_steps.append(LstmNode(self.lstm_param, lstm_state))

        idx = len(self.inputs) - 1
        if idx == 0:
            self.time_steps[idx].forward(x)
        else:
            prev_cell_state = self.time_steps[idx - 1].state.cell_state
            prev_hidden_state = self.time_steps[idx - 1].state.hidden_state
            self.time_steps[idx].forward(x, prev_cell_state, prev_hidden_state)
