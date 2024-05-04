#!/usr/bin/env python3

import numpy as np

# Make sure the results are reproducible
np.random.seed(0)

# Print helper function
def print_with_name(*args):
    for arg in args:
        print(f"""{arg[0]}: {arg[1]}""")

# Calculate the loss function (mean of the squared differences)
def loss_function_mse(inputs, results):
    # Calculate the mean of the squared differences
    squared_diff = np.square(inputs - results).sum(1)
    return squared_diff.mean()

# Number of dimensions of the input
D = 4
# Number of inputs
N = 1

weights_test = np.random.normal(size=(4,4))
bias_test = np.random.normal(size=(4,))

inputs = np.random.normal(size=(1,D))

print("### Results before training ###")
pre_output = inputs @ weights_test + bias_test
print_with_name(
    ("Inputs", inputs),
    ("Outputs", pre_output),
    ("Loss", loss_function_mse(inputs, pre_output))
)

# train loop
for epoch in range(1000):
    # Forward pass
    result = inputs @ weights_test + bias_test
    # Backward pass
    dL_dO = 2/N * (inputs - result)
    ## Single layer Jacobian
    dL_dW = inputs.T @ dL_dO
    dL_db = dL_dO.sum(0)
    # Update
    weights_test += 0.001 * dL_dW
    bias_test += 0.001 * dL_db

    outputs = inputs @ weights_test + bias_test
    print(f"Iteration {epoch} - MSE: {loss_function_mse(inputs, outputs)}")

outputs = inputs @ weights_test + bias_test

print("### Results after training ###")
print_with_name(
    ("Inputs", inputs),
    ("Outputs", outputs),
    ("Loss", loss_function_mse(inputs, outputs))
)