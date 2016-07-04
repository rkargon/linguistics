#!/usr/bin/python

"""
Library for utility functions, usually math related
"""
import math

PHI = (math.sqrt(5) - 1)/2


def lerp(a, b, r):
    """
    Linear interpolation of values
    :param a: Starting value
    :param b: Ending value
    :param r: Amount to interpolate between a and b.
        (When r is 0, returns a. When r is 1, return b)
    :return: a + r*(b - a)
    """
    return a + r*(b - a)


def golden_section_search(f, a, b, maximize=True, epsilon=1e-5):
    """
    Performs a golden section search to find the optimal value of a
     unimodal function within a certain interval.
    :param f: The function to optimize
    :param a: The lower part of the initial interval
    :param b: The higher part of the initial interval
    :param maximize: Whether to maximize or minimize the function.
     This function maximizes by default
    :param epsilon: The tolerance for different input values. By default 1e-5
    :return: The input value that gives the optimal value for the function
    """

    # ordering is: [a, c, d, b]
    c = b - PHI * (b - a)
    d = a + PHI * (b - a)

    if maximize:
        test = lambda: fc > fd
    else:
        test = lambda: fc < fd

    while abs(c - d) > epsilon:
        fc = f(c)
        fd = f(d)
        if test():
            b = d
            d = c
            c = b - PHI * (b - a)
        else:
            a = c
            c = d
            d = a + PHI * (b - a)
    return (b + a)/2.0
