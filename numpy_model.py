import numpy as np

# expected inputs and outputs

test_cases = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

# hyperparameters

input_size = 2
hidden_size = 2
learning_rate = 0.2
epochs = 100000

# initialize model with Xavier initialization

model = [
    np.random.randn(hidden_size, input_size + 1) / np.sqrt(input_size + 1),
    np.random.randn(hidden_size + 1) / np.sqrt(hidden_size + 1),
]

print(model)