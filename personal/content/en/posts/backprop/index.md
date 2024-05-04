+++
author = "Jose Lopes"
title = "Notes on Backpropagation"
date = "2024-05-04"
description = "Notes on backpropagation"
series = ["Themes Guide"]
aliases = ["migrate-from-jekyl"]
+++

Some notes on backpropagation.
<!--more-->

## Help links

Fast AI's [Lesson 13](https://course.fast.ai/Lessons/lesson13.html) by Jeremy
Howard goes into great detail to explain backpropagation.
To understand more about the mathematics behind backpropagation and how it works in pytorch, I recommend
[this](https://nasheqlbrm.github.io/blog/posts/2021-11-13-backward-pass.html#notation)
blog post (also recommended by Jeremy).


## Introduction

This blog post focus on backpropagation and on the optimization of a Single-Layer Perceptron (SLP).
The idea is to provide a minimum working example that can be used to understand the underlying concepts. Frameworks, such as Pytorch, provide a lot of functionality that abstracts way the details of training a neural network. Hopefully, this post will help to close the gap between theory and practice.

The full code used in this post can be found [here](#code).

# Single-Layer Perceptron

An SLP is a simple neural network that has one layer of neurons.

# Code
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


The complexity of the attention layer of the transformer scales linearly with the sequence length. This is
true for the Queries and Values matrices. However the number of attention weights has a quadratic dependence on
the sequence length, but it is independent of the length D of each input [book]


# Scaled dot-prodcut self-attention

The dot products in the attention computation can have large magnitudes and move
the arguments to the softmax function into a region where the largest value completely
dominates. Small changes to the inputs to the softmax function now have little effect on Problem 12.4 the output (i.e., the gradients are very small), making the model difficult to train. To
prevent this, the dot products are scaled by the square root of the dimension Dq of the
queries and keys (i.e., the number of rows in Ωq and Ωk, which must be the same): [book]

## H2
### H3
#### H4
##### H5
###### H6

## Paragraph

Xerum, quo qui aut unt expliquam qui dolut labo. Aque venitatiusda cum, voluptionse latur sitiae dolessi aut parist aut dollo enim qui voluptate ma dolestendit peritin re plis aut quas inctum laceat est volestemque commosa as cus endigna tectur, offic to cor sequas etum rerum idem sintibus eiur? Quianimin porecus evelectur, cum que nis nust voloribus ratem aut omnimi, sitatur? Quiatem. Nam, omnis sum am facea corem alique molestrunt et eos evelece arcillit ut aut eos eos nus, sin conecerem erum fuga. Ri oditatquam, ad quibus unda veliamenimin cusam et facea ipsamus es exerum sitate dolores editium rerore eost, temped molorro ratiae volorro te reribus dolorer sperchicium faceata tiustia prat.

Itatur? Quiatae cullecum rem ent aut odis in re eossequodi nonsequ idebis ne sapicia is sinveli squiatum, core et que aut hariosam ex eat.

## Blockquotes

The blockquote element represents content that is quoted from another source, optionally with a citation which must be within a `footer` or `cite` element, and optionally with in-line changes such as annotations and abbreviations.

#### Blockquote without attribution

> Tiam, ad mint andaepu dandae nostion secatur sequo quae.
> **Note** that you can use *Markdown syntax* within a blockquote.

#### Blockquote with attribution

> Don't communicate by sharing memory, share memory by communicating.<br>
> — <cite>Rob Pike[^1]</cite>

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.

## Tables

Tables aren't part of the core Markdown spec, but Hugo supports supports them out-of-the-box.

   Name | Age
--------|------
    Bob | 27
  Alice | 23

#### Inline Markdown within tables

| Italics   | Bold     | Code   |
| --------  | -------- | ------ |
| *italics* | **bold** | `code` |

## Code Blocks

#### Code block with backticks

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Example HTML5 Document</title>
</head>
<body>
  <p>Test</p>
</body>
</html>
```

#### Code block indented with four spaces

    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Example HTML5 Document</title>
    </head>
    <body>
      <p>Test</p>
    </body>
    </html>

#### Code block with Hugo's internal highlight shortcode
{{< highlight html >}}
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Example HTML5 Document</title>
</head>
<body>
  <p>Test</p>
</body>
</html>
{{< /highlight >}}

## List Types

#### Ordered List

1. First item
2. Second item
3. Third item

#### Unordered List

* List item
* Another item
* And another item

#### Nested list

* Fruit
  * Apple
  * Orange
  * Banana
* Dairy
  * Milk
  * Cheese

## Other Elements — abbr, sub, sup, kbd, mark

<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd><kbd>CTRL</kbd>+<kbd>ALT</kbd>+<kbd>Delete</kbd></kbd> to end the session.

Most <mark>salamanders</mark> are nocturnal, and hunt for insects, worms, and other small creatures.
