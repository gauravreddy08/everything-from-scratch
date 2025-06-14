# Backpropagation Derivation in a Neural Network


## Part 1: Backpropagation for the Final Layer (L)

### Goal

$$
\frac{\partial \mathcal{L}}{\partial W^L}
$$

### Step 1: Apply Chain Rule

$$
\frac{\partial \mathcal{L}}{\partial W^L} = \frac{\partial \mathcal{L}}{\partial a^L} \cdot \frac{\partial a^L}{\partial z^L} \cdot \frac{\partial z^L}{\partial W^L}
$$

### Step 2: Compute Each Term

#### Term 1 & 2: Loss and Activation Derivatives

For the final layer with sigmoid activation and binary cross-entropy loss:

$$
\frac{\partial \mathcal{L}}{\partial a^L} \cdot \frac{\partial a^L}{\partial z^L} = \hat{y} - y
$$

#### Term 3: Linear Combination Derivative

$$
z^L = W^L a^{L-1} + b^L
$$

$$
\frac{\partial z^L}{\partial W^L} = a^{L-1}
$$

### Final Result for Layer L

$$
\frac{\partial \mathcal{L}}{\partial W^L} = (\hat{y} - y) \cdot (a^{L-1})^T
$$


## Part 2: Backpropagation for Hidden Layers (L-1 → 0)


$$
\frac{\partial \mathcal{L}}{\partial W^{L-1}} = \frac{\partial \mathcal{L}}{\partial a^{L-1}} \cdot \frac{\partial a^{L-1}}{\partial z^{L-1}} \cdot \frac{\partial z^{L-1}}{\partial W^{L-1}}
$$

### Breaking Down Each Term

#### I. First Term: Error Propagation

$$
\frac{\partial \mathcal{L}}{\partial a^{L-1}} = \frac{\partial \mathcal{L}}{\partial z^L} \cdot \frac{\partial z^L}{\partial a^{L-1}} = (\hat{y} - y) \cdot W^L
$$


#### II. Second Term: Activation Derivative

For ReLU activation in hidden layers:

$$
\frac{\partial a^{L-1}}{\partial z^{L-1}} = \text{ReLU}'(z^{L-1})
$$

$$
\text{ReLU}'(z) = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{otherwise}
\end{cases}
$$

#### III. Third Term: Linear Combination

$$
\frac{\partial z^{L-1}}{\partial W^{L-1}} = (a^{L-2})^T
$$

### Final Result for Hidden Layers

$$
\frac{\partial \mathcal{L}}{\partial W^{L-1}} = \left( W^L \right)^T (\hat{y} - y) \odot \text{ReLU}'(z^{L-1}) \cdot (a^{L-2})^T
$$
