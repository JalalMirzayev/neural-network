import numpy


def random_matrix(rows: int, columns: int) -> numpy.array:
    return numpy.random.rand(rows, columns) - 0.5 * numpy.ones((rows, columns))


def relu(x: numpy.array) -> numpy.array:
    return x * (x > 0)


def sigmoid(x: numpy.array) -> numpy.array:
    exponential = numpy.exp(x)
    return exponential / (1 + exponential)


def softmax(x: numpy.array) -> numpy.array:
    x_shifted = x - numpy.max(x)
    x_exponential = numpy.exp(x_shifted)
    return x_exponential / numpy.sum(x_exponential)


def pre_activation(weight_matrix: numpy.array, activation: numpy.array, bias: numpy.array) -> numpy.array:
    return numpy.matmul(weight_matrix, activation) + bias


def activation(pre_activation: numpy.array, activation_function: callable) -> numpy.array:
    return activation_function(pre_activation)


if __name__ == '__main__':
    pass
