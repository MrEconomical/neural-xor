import math
import random

# expected XOR inputs and outputs

xor_values = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

# hyperparameters

input_size = 2
hidden_size = 2
learning_rate = 0.01

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

def forward(input: list[float]) -> float:
    input.append(1) # add bias
    hidden_result = []
    for weights in model[0]:
        result = 0
        for w in range(len(weights)):
            result += weights[w] * input[w]
        hidden_result.append(sigmoid(result))

    hidden_result.append(1) # add bias
    output_result = 0
    for w in range(len(model[1])):
        output_result += model[1][w] * hidden_result[w]
    
    return sigmoid(output_result)

for case in xor_values:
    print("testing", case)
    result = forward(case[0])
    print(result)