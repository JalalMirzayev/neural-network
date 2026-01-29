import pytest
import numpy
from utils import random_matrix
from utils import relu
from utils import softmax

@pytest.mark.parametrize("rows, columns, expected_shape", [
    (2, 3, (2, 3)),
    (0, 0, (0, 0)),
    (2, 2, (2, 2))
])
def test_random_matrix_shape(rows, columns, expected_shape):
    assert random_matrix(rows, columns).shape == expected_shape

@pytest.mark.parametrize("rows, columns, min, max", [
    (100, 200, -0.5, 0.5)
])
def test_random_matrix_range(rows, columns, min, max):
    actual_matrix = random_matrix(rows, columns)
    assert min <= actual_matrix.min()
    assert max > actual_matrix.max()

relu_test_matrix: numpy.array = numpy.array([[0.12, -0.11, 0.89], [0.98, 0.01, -0.13]])

def test_relu():
    actual = relu(relu_test_matrix)
    assert numpy.array_equal(actual, numpy.array([[0.12, 0, 0.89], [0.98, 0.01, 0]]))

def test_relu_shape_invariance():
    actual = relu(relu_test_matrix)
    assert actual.shape == (2, 3)

def test_relu_range():
    actual = relu(relu_test_matrix)
    assert actual.min() == 0
    assert actual.max() == 0.98

@pytest.mark.parametrize("input_vector, expected", [
    (numpy.array([1, 2, 3]), numpy.array([
        numpy.exp(-2)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(-1)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(0)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0))])),
    (numpy.array([[1, 2, 3]]), numpy.array([[
        numpy.exp(-2)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(-1)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(0)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0))]])),
    (numpy.array([[1, 1, 1]]), numpy.array([[1/3, 1/3, 1/3]])),
    (numpy.array([[1, -numpy.inf, -numpy.inf]]), numpy.array([[1, 0, 0]])),
    (
        numpy.array([[100000, 0, 0]]), 
        numpy.array([[
            numpy.exp(0)/(numpy.exp(0) + 2 * numpy.exp(-10000)),
            numpy.exp(-10000)/(numpy.exp(0) + 2 * numpy.exp(-10000)),
            numpy.exp(-10000)/(numpy.exp(0) + 2 * numpy.exp(-10000))]])),
])
def test_softmax(input_vector, expected):
    actual = softmax(input_vector)
    assert numpy.array_equal(actual, expected)


@pytest.mark.parametrize("input_vector, expected", [
    (numpy.array([1, 2, 3]), numpy.array([
        numpy.exp(-2)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(-1)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(0)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0))])),
    (numpy.array([[1, 2, 3]]), numpy.array([[
        numpy.exp(-2)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(-1)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0)),
        numpy.exp(0)/(numpy.exp(-2) + numpy.exp(-1) + numpy.exp(0))]])),
    (numpy.array([[1, 1, 1]]), numpy.array([[1/3, 1/3, 1/3]])),
    (numpy.array([[1, -numpy.inf, -numpy.inf]]), numpy.array([[1, 0, 0]])),
    (
        numpy.array([[100000, 0, 0]]), 
        numpy.array([[
            numpy.exp(0)/(numpy.exp(0) + 2 * numpy.exp(-10000)),
            numpy.exp(-10000)/(numpy.exp(0) + 2 * numpy.exp(-10000)),
            numpy.exp(-10000)/(numpy.exp(0) + 2 * numpy.exp(-10000))]])),
    (numpy.array([1]), 1)
])
def test_softmax(input_vector, expected):
    actual = softmax(input_vector)
    assert numpy.array_equal(actual, expected)


def test_softmax():
    rows = numpy.random.randint(1, 10000 + 1)
    columns = numpy.random.randint(1, 10000 + 1)
    random_input = numpy.random.rand(rows, columns)
    actual = softmax(random_input)
    assert numpy.abs(numpy.sum(actual) - 1) < 0.000001
    assert 0 <= actual.min()
    assert 1 >= actual.max()