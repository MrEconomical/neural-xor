import math
import random

# expected inputs and outputs

test_cases = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

# hyperparameters

input_size = 2
hidden_size = 4
learning_rate = 0.5
epochs = 10000

# initialize model with random weights

def get_random_weight() -> float:
    return (random.random() - 0.5) * 4

model = [[], []]
for n in range(hidden_size):
    weights = []
    for w in range(input_size + 1):
        weights.append(get_random_weight())
    model[0].append(weights)
for n in range(hidden_size + 1):
    model[1].append(get_random_weight())

# activation function

def sigmoid(value: int):
    return 1 / (1 + math.exp(-value))

# calculate forward propagation result

def forward(input: list[float]) -> (list[float], float):
    hidden_result = []
    for weights in model[0]:
        result = 0
        for w in range(len(input)):
            result += weights[w] * input[w]
        result += weights[-1] # add bias
        hidden_result.append(sigmoid(result))

    output = 0
    for w in range(len(hidden_result)):
        output += model[1][w] * hidden_result[w]
    output += model[1][-1] # add bias
    
    return hidden_result, sigmoid(output)

# update weights in back propagation

def back_prop(input: list[float], hidden_result: list[float], output: float, expected: float) -> float:
    # calculate output error and delta for output neuron

    output_error = 0.5 * (expected - output) ** 2
    output_error_derivative = output - expected # derivative of error function
    output_sigmoid_derivative = output * (1 - output) # derivative of sigmoid
    output_delta = output_error_derivative * output_sigmoid_derivative # derivative of error with respect to pre-sigmoid output

    # calculate updated output neuron weights

    output_weights = []
    for w in range(len(hidden_result)):
        weight_derivative = output_delta * hidden_result[w] # derivative of error with respect to weight
        new_weight = model[1][w] - learning_rate * weight_derivative # update weight to reduce error
        output_weights.append(new_weight)
    
    new_bias_weight = model[1][-1] - learning_rate * output_delta # bias is a fixed input of 1
    output_weights.append(new_bias_weight)

    # back propagate error to hidden layer

    hidden_weights = [[]] * len(model[0])
    for n in range(len(model[0])):
        hidden_error_derivative = output_delta * model[1][n] # derivative of error with respect to hidden neuron output
        hidden_sigmoid_derivative = hidden_result[n] * (1 - hidden_result[n]) # derivative of sigmoid
        hidden_delta = hidden_error_derivative * hidden_sigmoid_derivative # derivative of error with respect to pre-sigmoid output

        for w in range(len(input)):
            weight_derivative = hidden_delta * input[w] # derivative of error with respect to weight
            new_weight = model[0][n][w] - learning_rate * weight_derivative # update weight to reduce error
            hidden_weights[n].append(new_weight)
        
        new_bias_weight = model[0][n][-1] - learning_rate * hidden_delta # bias is a fixed input of 1
        hidden_weights[n].append(new_bias_weight)

    # set model weights

    model[0] = hidden_weights
    model[1] = output_weights

    # return output error

    return output_error

# train over epochs

log_interval = epochs // 20
for e in range(epochs):
    total_error = 0
    for case in test_cases:
        hidden_result, output = forward(case[0])
        output_error = back_prop(case[0], hidden_result, output, case[1])
        total_error += output_error
    if e % log_interval == 0:
        print("epoch", e, "mean error", total_error / 4)

# evaluate model

for case in test_cases:
    h, output = forward(case[0])
    print(case[0], "expected output:", case[1], "model output:", output)