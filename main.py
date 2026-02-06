import numpy
from utils import random_matrix
from utils import relu
from utils import sigmoid
from utils import activation
from utils import pre_activation


def main():
    # Initialize costs
    costs = []
    error_tolerance = 0.001
    # Initialize learning rate
    eta = 0.1
    # Initialize weights
    weight_matrix_1 = random_matrix(rows=2, columns=3)
    weight_matrix_2 = random_matrix(rows=2, columns=2)
    bias_1 = random_matrix(rows=2, columns=1)
    bias_2 = random_matrix(rows=2, columns=1)

    keep_training = True
    epoch = 1

    while keep_training:

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}; cost = {current_cost}')
        epoch = epoch + 1

        # Choose target values such that sigmoid can work between 0 and 1.
        samples = [
            (numpy.array([[1], [2], [3]]), numpy.array([[0.1], [1]])),
            (numpy.array([[1], [3], [3]]), numpy.array([[0.1], [0.1]])),
            ]
        
        # Create lists to store weights per sample
        grad_weights_1_samples = []
        grad_bias_1_samples = []

        grad_weights_2_samples = []
        grad_bias_2_samples = []
        current_cost = 0

        for sample in samples:
            x_input, y_target = sample

            # Calculate activations using 
            # Initial activation
            activation_0: numpy.array = numpy.array(x_input)
            # z^(l) = W^(l)a^(l-1) + b^(l)
            # a^(l) = f_\text{point-wise}(z^(l))
            pre_activation_1 = pre_activation(weight_matrix=weight_matrix_1, activation=activation_0, bias=bias_1)
            activation_1 = activation(pre_activation_1, activation_function=sigmoid)

            pre_activation_2 = pre_activation(weight_matrix=weight_matrix_2, activation=activation_1, bias=bias_2)
            activation_2 = activation(pre_activation_2, activation_function=sigmoid)
            
            # Calculate sample and aggregated cost
            cost = ((activation_2 - y_target).T @ (activation_2 - y_target))[0, 0]
            current_cost = current_cost + cost

            # Delta output of last layer
            delta_2: numpy.array = 2 * (activation_2 - y_target) * activation_2 * (1 - activation_2)

            # Delta for previous layers
            delta_1: numpy.array = (weight_matrix_2.T @ delta_2) * activation_1 * (1 - activation_1)
            delta_0: numpy.array = (weight_matrix_1.T @ delta_1) * activation_0 * (1 - activation_0)

            # Gradient of weights using
            # del C(sample) / del W^(l) = delta^(l)[a^(l-1)]^T
            grad_weights_2 = delta_2 @ activation_1.T
            grad_weights_1 = delta_1 @ activation_0.T
            # del C(sample) / del b^(l) = delta^(l)
            grad_bias_2 = delta_2
            grad_bias_1 = delta_1

            grad_weights_2_samples.append(grad_weights_2)
            grad_weights_1_samples.append(grad_weights_1)
            grad_bias_2_samples.append(grad_bias_2)
            grad_bias_1_samples.append(grad_bias_1)

        costs.append(current_cost)
        if current_cost < error_tolerance:
            keep_training = False
            print(f'epoch: {epoch}, cost: {current_cost}')

        # Average gradients over all samples
        # del C / del W^(l) = 1 / n * sum_x del C(sample=x) / del W^(l)
        # del C / del b^(l) = 1 / n * sum_x del C(sample=x) / del Wb^(l)
        grad_weights_2_average = numpy.mean(grad_weights_2_samples, axis=0)
        grad_weights_1_average = numpy.mean(grad_weights_1_samples, axis=0)
        grad_bias_2_average = numpy.mean(grad_bias_2_samples, axis=0)
        grad_bias_1_average = numpy.mean(grad_bias_1_samples, axis=0)

        # Update weights and biases with gradient decent algorithm
        weight_matrix_2 = weight_matrix_2 - eta * grad_weights_2_average
        weight_matrix_1 = weight_matrix_1 - eta * grad_weights_1_average
        bias_2 = bias_2 - eta * grad_bias_2_average
        bias_1 = bias_1 - eta * grad_bias_1_average


if __name__ == "__main__":
    main()
