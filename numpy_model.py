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

# activation function

def sigmoid(value) -> float:
    return 1 / (1 + np.exp(-value))

# calculate forward propagation result

def forward(input: list[float]):
    hidden_result = np.dot(model[0][:, :-1], input) + model[0][:, -1:].flat # multiply hidden weights by input and add bias
    hidden_output = sigmoid(hidden_result) # apply activation function to result
    output_result = np.dot(model[1][:-1], hidden_output) + model[1][-1] # multiply output weights by hidden output and add bias
    return hidden_output, sigmoid(output_result)

hidden_output, output = forward(test_cases[0][0])
print(hidden_output, output)