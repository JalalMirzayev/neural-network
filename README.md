# Mathematical Representation of Neural Networks

Assume a very simple situation with an input $\mathbf{x} \in \mathbb{R}^3$, two hidden layers with two nodes each and a softmax to produce the output $\hat{y}$. As $\mathbf{x}$ can be seen as an input activation we will reference it as $\mathbf{x}=\mathbf{a}^{(0)}$.

We will distinguish between pre-activation $\mathbf{z}^{(l)}$ and the activation $\mathbf{a}^{(l)}=\mathbf{f}(\mathbf{z}^{(L)})$ which applies the activation function to the pre-activation.

For the pre-activation at the first hidden layer we have

$$
\begin{bmatrix}z_0^{(1)}\\z_1^{(1)}
\end{bmatrix}=
\begin{bmatrix}
    w_{00}^{(1)} & w_{01}^{(1)} & w_{02}^{(1)}\\
    w_{10}^{(1)} & w_{11}^{(1)} & w_{12}^{(1)}\\
\end{bmatrix}
\begin{bmatrix}
a_0^{(0)}\\
a_1^{(0)}\\
a_2^{(0)}
\end{bmatrix} +
\begin{bmatrix}
b_0^{(1)}\\
b_1^{(1)}
\end{bmatrix}
$$

We can simply put this into a matrix formulation

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{a}^{(0)} + \mathbf{b}^{(1)}.$$

Using this formulation we can note a general relationship for the format of these matrices and vectors.

- $\mathbf{z}^{(l)} \in \mathbb{R}^{n_l}$: The pre-activation vector has a dimensionality which is equal to the number of neurons $n_l$ in layer $l$.
- $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$: The weight matrix has $n_l$ (number of neurons in current layer) rows and $n_{l-1}$ (number of neurons in previous layer because they will create the activations) columns.
- $\mathbf{a}^{(l)} \in \mathbb{R}^{n_{l}}$: Because every neuron of the current layer $l$ will have a certain activation.
- $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$: Every neuron in the layer $l$ has a bias associated with it.

## Applying the activation function

We require a nonlinear activation function which breaks linearity. If we would not use this nonlinearity the overall transfer behavior from inputs to outputs could be represented by a linear function and the neural network would loose it's flexibility. Assuming a nonlinear activation function $\mathbf{f}$ we obtain:

$$\mathbf{a}^{L} = \mathbf{f}(\mathbf{z}^{L}).$$

# Forward pass

We are ready to formulate the forward pass algorithm. Assume an input $\mathbf{a}^{(0)}$ and $L_\text{end}$ layers in the network.

- 1: Initialize Weights and Biases: $\mathbf{W}^{l}$ and $\mathbf{b}^{l}$ for all $1 \leq l \leq l_\text{end}$
- Initialize the cost with $C = 0$
- 2: Loop over all samples $\{(\mathbf{x}_1=\mathbf{a}^{(0)}_1, \mathbf{y}_\text{1, target}=\mathbf{a}^{(l_\text{end})}_1),\ldots, (\mathbf{x}_N = \mathbf{a}^{(0)}_N, \mathbf{y}_\text{N, target} =\mathbf{a}^{(l_\text{end})}_N)\}$
  - 1 Initialize $\mathbf{x}_i = \mathbf{a}^{(0)}$.
    for
  - 2: Loop for $l$ in $\{1, 2, \ldots, l_\text{end} \}$
    - 1: Calculate pre-activation (weighted input) of layer $l$ from
      $$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}.$$
    - 2: Calculate activation of layer $l$ from
      $$\mathbf{a}^{(l)} = \mathbf{f}(\mathbf{z}^{(l)}).$$
    - 3: Depending on the task at hand (regression vs. classification) you might need to apply softmax to the activation of the last layer $\mathbf{a}^{(L_\text{end})}$ which gives us the output
      $$
      \mathbf{y}_\text{i, predicted} = \mathbf{a}^{(l_\text{end})} =  \begin{cases}
          \mathbf{f}(\mathbf{z}^{(l_\text{end})}) & \text{if Regression}\\
          \text{softmax}(\mathbf{z}^{(l_\text{end})}) & \text{if Classification}
      \end{cases}
      $$
  - 3: Calculate the squared difference between the predicted and the actual output value.
    $$C_i = (y_\text{i, predicted} - y_\text{i, target})^2$$
  - 4: Update cost with $C \rightarrow C = C + C_i$
- 8: Go to step 2 and repeat the process for the next training sample.
- 9: Determine the average squared error for all samples.

# Gradient Descent with Back-Propagation

1. Initialize Network parameters (weights, biases) with random values
2. For each training sample $\{(\mathbf{x}_1, \mathbf{y}_\text{1, target}),\ldots, (\mathbf{x}_N, \mathbf{y}_\text{N, target})\}$:

   a) Compute activations $a^{(l)}$ for the entire network using
   $$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}.$$
   $$\mathbf{a}^{(l)} = \mathbf{f}(\mathbf{z}^{(l)}).$$
   For last layer $\mathbf{a}^{(l_\text{end})}$

   $$
      \mathbf{y}_\text{i, predicted} =  \begin{cases}
          \mathbf{a}^{(l_\text{end})} & \text{if Regression}\\
          \text{softmax}(\mathbf{a}^{(l_\text{end})}) & \text{if Classification}
      \end{cases}
   $$

   b) Compute $\mathbf{\delta}^{(l_\text{end})} = 2 (\mathbf{a}^{(l_\text{end})} - \mathbf{a}^{(l_\text{end})}_\text{target}) \odot \mathbf{f}'(\mathbf{z}^{(l)}),$ in which $\odot$ is the point-wise Hadamard product and $\mathbf{f}'(\mathbf{z}^{(l)})$ is derivative of the activation function which is applied to every coordinate of $\mathbf{z}^{(l)}$.

   c) Compute all other deltas with
   $$\mathbf{\delta}^{(l)} = \left[\mathbf{W}^{(l+1)} \right]^T\mathbf{\delta}^{(l+1)} \odot \mathbf{f}'(\mathbf{z}^{(l)})$$

   d) Compute sample gradient for the sample cost $C_x$ with respect to the weights $w^{(l)}_{jk}$ and biases $b^{(l)}_j$

   Component representation
   $$\dfrac{\partial C_x}{\partial w^{(l)}_{jk}} = \delta^{(l)}_j a^{(l-1)}_k$$
   $$\dfrac{\partial C_x}{\partial b^{(l)}_j} = \delta^{(l)}_j$$

   Vector/Matrix representation:
   $$\dfrac{\partial C_x}{\partial \mathbf{W}^{(l)}} = \mathbf{\delta}^{(l)}\left[\mathbf{a}^{(l-1)}\right]^T$$
   $$\dfrac{\partial C_x}{\partial \mathbf{b}^{(l)}} = \mathbf{\delta}^{(l)}$$

3. Average over all samples
   $$\dfrac{\partial C}{\partial \mathbf{W}^{(l)}} = \dfrac{1}{N} \sum_{x}\dfrac{\partial C_x}{\partial \mathbf{W}^{(l)}}$$
   $$\dfrac{\partial C}{\partial \mathbf{b}^{(l)}} = \dfrac{1}{N} \sum_{x}\dfrac{\partial C_x}{\partial \mathbf{b}^{(l)}} $$

4. Update weights and biases
   $$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \dfrac{\partial C}{\partial \mathbf{W}^{(l)}} $$
  $$\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \dfrac{\partial C}{\partial \mathbf{b}^{(l)}} $$

5. Repeat steps 2 - 4 until Cost is reduces below a acceptable level or iterate over a fixed number of epochs.

# References

- [Shree Nayar - Backpropagation Algorithm | Neural Networks](https://www.youtube.com/watch?v=sIX_9n-1UbM)
