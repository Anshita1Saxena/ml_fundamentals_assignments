import numpy as np


def make_array_from_list(some_list):
    arr = np.asarray(some_list)
    return arr


def make_array_from_number(num):
    num_a = [num]
    num_arr = np.array(num_a)
    return num_arr


class NumpyBasics:
    def add_arrays(self, a, b):
        c = np.add(a, b)
        return c

    def add_array_number(self, a, num):
        # Call function to make the number an array
        new_arr = make_array_from_number(num)
        # Broadcast concept
        c = np.add(a, new_arr)
        return c

    def multiply_elementwise_arrays(self, a, b):
        c = np.multiply(a, b)
        return c

    def dot_product_arrays(self, a, b):
        c = np.dot(a, b)
        return c

    def dot_1d_array_2d_array(self, a, m):
        # consider the 2d array to be like a matrix
        c = np.dot(a, m)
        return c
