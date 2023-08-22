import math
import random

# expected XOR inputs and outputs

xor_values = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

# hyperparameters

input_size = 2
hidden_size = 2
learning_rate = 0.5

# initialize model with random weights

def get_random_weight() -> float:
    return (random.random() - 0.5) * 2

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

def back_prop(input: list[float], hidden_result: list[float], output: float, expected: float):
    # calculate output error and delta for output neuron

    output_error = 0.5 * (expected - output) ** 2
    print("output error:", output, expected, output_error)
    output_error_derivative = output - expected # derivative of error function
    output_sigmoid_derivative = output * (1 - output) # derivative of sigmoid
    output_delta = output_error_derivative * output_sigmoid_derivative # derivative of error with respect to pre-sigmoid output
    print("output delta:", output_delta)

    # calculate updated output neuron weights
    output_weights = []
    for w in range(len(hidden_result)):
        weight_derivative = output_delta * hidden_result[w] # derivative of error with respect to weight
        new_weight = model[1][w] - learning_rate * weight_derivative # update weight to reduce error
        output_weights.append(new_weight)
    new_bias_weight = model[1][-1] - learning_rate * output_delta # bias is a fixed input of 1
    output_weights.append(new_bias_weight)

    print("current_output_weights:", model[1])
    print("adjusted output weights:", output_weights)

    # test the new weights for sanity
    print()
    print("old result:", output)
    new_result = sigmoid(hidden_result[0] * output_weights[0] + hidden_result[1] * output_weights[1] + output_weights[2])
    print("new result:", new_result)

    model[1] = output_weights

for x in range(1):
    for case in [xor_values[3]]:
        print("testing", case)
        hidden_result, output = forward(case[0])
        print(hidden_result, output)
        print()
        back_prop(case[0], hidden_result, output, case[1])