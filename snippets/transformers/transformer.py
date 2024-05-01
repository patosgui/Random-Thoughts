#!/usr/bin/env python3

import numpy as np

# Number of dimensions of each input
D = 4
# Number of inputs
N = 3

np.random.seed(0)

def softmax(x):
    """
    Computes the softmax function.

    Args:
        x (np.ndarray): The input to the softmax function.

    Returns:
        np.ndarray: The output of the softmax function.
    """

    return np.exp(x)/(np.exp(x).sum(-1))

# TODO: Check if this is correct vs numpy version
def layer_norm(x, gamma, beta, eps=1e-5):
    # Calculate the mean and variance across the features (last axis)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize the input
    normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Scale and shift the normalized input
    output = gamma * normalized + beta
    
    return output


def generate_network(dim):
    """
    Generates a simple neural network with randomly initialized weights and biases.
    
    Args:
        D (int): The dimensionality of the input and output.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The randomly initialized weights and biases for the network.
    """

    weights = np.random.normal(size=(dim,dim))
    bias = np.random.normal(size=(dim,))
    return weights, bias

def self_attention(num, dim, inputs):
    weights_q, bias_q = generate_network(dim)
    weights_k, bias_k = generate_network(dim)
    weights_v, bias_v = generate_network(dim)

    inputs = np.random.normal(size=(num,dim))

    # Calculate attentions
    queries = inputs @ weights_q + bias_q # [3x4]
    keys = inputs @ weights_k + bias_k # [3x4]

    ## Multiply every query vector by every key vector
    ### Every column tells us how much attention each input should pay to all others
    weights_a = queries @ keys.T # [3x3]
    scaled_weights_a = softmax(weights_a / np.sqrt(dim)) # [3x3]

    # Calculate values
    values = inputs @ weights_v + bias_v # [3x4]

    # Calculate output
    # Store the values in the columns of the "values" matrix so that we
    # can multiply them with the attention weights, where each column maps to a word
    outputs = scaled_weights_a @ values # [4x3] @ [3x3] = [4x3]

    print(
    f"""
Queries: 
{queries}
Keys: 
{keys}
Values:
{values}
Attentions:
{scaled_weights_a}
Outputs:
{outputs}
    """
    )

    return outputs

inputs = np.random.normal(size=(N,D))
print(
f"""
Inputs 
{inputs}
""")

self_attention_1 = self_attention(num=N, dim=D//2, inputs=inputs)
self_attention_2 = self_attention(num=N, dim=D//2, inputs=inputs)

# TODO: How to concatenate multi heads? Is therea any multiplication required?
result = np.concatenate((self_attention_1, self_attention_2), axis=1)

print(
f"""
Multi Head Result
{result}
"""
)

# TODO: Can we use the defaults values of the parameters?
result = layer_norm(result, gamma=1, beta=0)

print(
f"""
Layer Norm
{result}
"""
)

# Generate as many networks as inputs
parallel_nns = [generate_network(dim=D) for _ in range(N)]

for input, nn in zip(result, parallel_nns):
    print("üüü")
    print(input @ nn[0] + nn[1])

# Calculate the output of each parallel network
result = np.concatenate([[input @ nn[0] + nn[1] for input, nn in zip(result, parallel_nns)]], axis=1)
print(
f"""
Result after parallel networks
{result}
"""
)

result = layer_norm(result, gamma=1, beta=0)

print(
f"""
Layer Norm
{result}
"""
)

# TODO: Residual connection