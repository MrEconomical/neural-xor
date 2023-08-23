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

# activation function

def sigmoid(value) -> float:
    return 1 / (1 + np.exp(-value))

# calculate forward propagation result

def forward(input):
    hidden_result = np.dot(model[0][:, :-1], input) + model[0][:, -1:].flat # multiply hidden weights by input and add bias
    hidden_output = sigmoid(hidden_result) # apply activation function to result
    output_result = np.dot(model[1][:-1], hidden_output) + model[1][-1] # multiply output weights by hidden output and add bias
    return hidden_output, sigmoid(output_result)

# update weights in back propagation

def back_prop(input, hidden_output, output, expected):
    # calculate output error and delta for output layer

    output_error = 0.5 * (expected - output) ** 2
    output_error_derivative = output - expected # derivative of error function
    output_sigmoid_derivative = output * (1 - output) # derivative of sigmoid
    output_delta = output_error_derivative * output_sigmoid_derivative # derivative of error with respect to pre-sigmoid output

    # calculate updated output neuron weights

    output_weight_derivatives = np.concatenate((output_delta * hidden_output, [output_delta])) # derivative of error with respect to weights
    new_output_weights = model[1] - learning_rate * output_weight_derivatives # update output weights to reduce error

    # back propagate error to hidden layer

    hidden_error_derivatives = output_delta * model[1][:-1] # derivative of error with respect to hidden neuron output
    hidden_sigmoid_derivatives = hidden_output * (1 - hidden_output) # derivative of hidden sigmoid output
    hidden_deltas = hidden_error_derivatives * hidden_sigmoid_derivatives # derivative of error with respect to pre-sigmoid outputs

    hidden_weight_derivatives = np.concatenate(
        (
            np.multiply.outer(hidden_deltas, input),
            np.reshape(hidden_deltas, (hidden_deltas.size, 1))
        ),
        axis=1,
    )
    new_hidden_weights = model[0] - learning_rate * hidden_weight_derivatives # update hidden weights to reduce error

    # set model weights

    model[0] = new_hidden_weights
    model[1] = new_output_weights

    # return output error

    return output_error

# train over epochs

log_interval = epochs // 20
for e in range(epochs):
    total_error = 0
    for case in test_cases:
        hidden_output, output = forward(case[0])
        output_error = back_prop(case[0], hidden_output, output, case[1])
        total_error += output_error
    if e % log_interval == 0:
        print("epoch", e, "mean error", total_error / 4)

# evaluate model

print(model)
for case in test_cases:
    h, output = forward(case[0])
    print(case[0], "expected output:", case[1], "model output:", output)