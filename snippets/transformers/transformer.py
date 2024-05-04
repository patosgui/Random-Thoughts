#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import Embedding
from transformers import AutoTokenizer

# Number of dimensions of each input
D = 4
# Number of inputs
N = 3

np.random.seed(0)

def mse(inputs, results):
    # Calculate the mean of the squared differences
    squared_diff = np.square(inputs - results).sum(1)
    return squared_diff.mean()


def print_with_name(*args):
    for arg in args:
        print(f"""
{arg[0]}: {arg[1]}
""")

# from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
 
P = getPositionEncoding(seq_len=4, d=4, n=100)

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

    print_with_name(
        ("Queties", queries),
        ("Keys", keys),
        ("Values", values),
        ("Attentions", scaled_weights_a),
        ("Outputs", outputs)
    )

    return outputs

inputs = np.random.normal(size=(1,D))
weights_test, bias_test = generate_network(4)

# train loop
# Based on https://nasheqlbrm.github.io/blog/posts/2021-11-13-backward-pass.html#notation
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
    print(f"Iteration {epoch} - MSE: {mse(inputs, outputs)}")


print(inputs)
outputs = inputs @ weights_test + bias_test
print(outputs)

exit(1)
result = inputs @ weights_test + bias_test
dL_dO = 2/N * (inputs - result)
dL_dW = inputs.T @ dL_dO
print(dL_dW)
exit(1)
dL_db = dL_dO.sum(0)
print(dJ_dO)

exit(1)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = "You can now install TorchText using pip!"
input_tokens = tokenizer.tokenize(inputs)
input_token_ids = tokenizer.convert_tokens_to_ids(input_tokens)
seq_len = len(input_token_ids)

embeddings = Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=D)
encoding = getPositionEncoding(seq_len=seq_len, d=D)

with torch.no_grad():
    inputs = embeddings(torch.tensor(input_token_ids)) + encoding

print("## Input Handling ##")
print_with_name(
    ("Input Tokens", input_tokens),
    ("Input Token IDs", input_token_ids),
    ("Sequence Length", seq_len),
    ("Input With Positional Encoding", inputs),
)



print("## Transformer ##")
print("### Multi Head Attention ###")

self_attention_1 = self_attention(num=seq_len, dim=D//2, inputs=inputs[:,:D//2])
self_attention_2 = self_attention(num=seq_len, dim=D//2, inputs=inputs[:,D//2:])

# TODO: How to concatenate multi heads? Is therea any multiplication required?
result = np.concatenate((self_attention_1, self_attention_2), axis=1)

print_with_name(
    ("Self Attention 1", self_attention_1),
    ("Self Attention 2", self_attention_2),
    ("Multi Head Result", result),
)

# TODO: Can we use the defaults values of the parameters?
result = layer_norm(result, gamma=1, beta=0)

print("### Layer Norm ###")
print_with_name(
    ("Layer Norm", result),
)

# Generate as many networks as inputs
parallel_nns = [generate_network(dim=D) for _ in range(seq_len)]

for input, nn in zip(result, parallel_nns):
    print(input @ nn[0] + nn[1])

# Calculate the output of each parallel network
result = np.concatenate([[input @ nn[0] + nn[1] for input, nn in zip(result, parallel_nns)]], axis=1)


print("### MLP ###")
print_with_name(
    ("Result after parallel networks", result),
)

result = layer_norm(result, gamma=1, beta=0)

print("### Layer Norm ###")
print_with_name(
    ("Layer Norm", result),
)


print("debug")
print(inputs)
print(result)

# Calculate the mean of the squared differences
squared_diff = np.square(inputs - result).sum(1)
mse = squared_diff.mean()

J_activation = 1/seq_len * (inputs - result)

print(J_activation)
exit(1)

print("Mean Squared Error (MSE):", mse)

# Backpropagation https://nasheqlbrm.github.io/blog/posts/2021-11-13-backward-pass.html
# TODO: Residual connection