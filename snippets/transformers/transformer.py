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

def generate_network(D):
    """
    Generates a simple neural network with randomly initialized weights and biases.
    
    Args:
        D (int): The dimensionality of the input and output.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The randomly initialized weights and biases for the network.
    """

    weights = np.random.normal(size=(D,D))
    bias = np.random.normal(size=(D,))
    return weights, bias

def self_attention():
    weights_q, bias_q = generate_network(D)
    weights_k, bias_k = generate_network(D)
    weights_v, bias_v = generate_network(D)

    inputs = np.random.normal(size=(N,D))

    # Calculate attentions
    queries = inputs @ weights_q + bias_q # [3x4]
    keys = inputs @ weights_k + bias_k # [3x4]

    ## Multiply every query vector by every key vector
    ### Every column tells us how much attention each input should pay to all others
    weights_a = queries @ keys.T # [3x3]
    scaled_weights_a = softmax(weights_a) # [3x3]

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


self_attention_1 = self_attention()
self_attention_2 = self_attention()
