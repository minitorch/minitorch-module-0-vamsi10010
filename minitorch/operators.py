"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        The product x * y

    """
    return x * y


def id(x: float) -> float:
    """Identity function that returns its input unchanged.

    Args:
    ----
        x: Input number

    Returns:
    -------
        The same number x

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        The sum x + y

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: Input number

    Returns:
    -------
        The negation -x

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        True if x < y, False otherwise

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        True if x == y, False otherwise

    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        The larger of x and y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are approximately equal.

    Uses tolerance of 1e-2 for comparison: |x - y| < 1e-2

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        True if numbers are close, False otherwise

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid activation function.

    Uses numerically stable computation:
    - If x >= 0: 1 / (1 + exp(-x))
    - If x < 0: exp(x) / (1 + exp(x))

    Args:
    ----
        x: Input number

    Returns:
    -------
        Sigmoid of x, a value between 0 and 1

    """
    return (1 / (1 + math.exp(-x))) if x >= 0 else (math.exp(x) / (1 + math.exp(x)))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) activation function.

    Args:
    ----
        x: Input number

    Returns:
    -------
        max(0, x)

    """
    return x if x > 0 else 0


def log(x: float) -> float:
    """Compute the natural logarithm.

    Args:
    ----
        x: Input number (must be positive)

    Returns:
    -------
        Natural logarithm of x

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
    ----
        x: Input number

    Returns:
    -------
        e raised to the power of x

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the multiplicative inverse (reciprocal).

    Args:
    ----
        x: Input number

    Returns:
    -------
        1/x if x != 0, otherwise infinity

    """
    return 1 / x if x != 0 else math.inf


def log_back(x: float, y: float) -> float:
    """Compute the derivative of log with respect to x, scaled by y.

    This is the backward pass for the log function.

    Args:
    ----
        x: Input to the original log function
        y: Gradient from the downstream computation

    Returns:
    -------
        (1/x) * y

    """
    return inv(x) * y


def inv_back(x: float, y: float) -> float:
    """Compute the derivative of inv with respect to x, scaled by y.

    This is the backward pass for the inv function.

    Args:
    ----
        x: Input to the original inv function
        y: Gradient from the downstream computation

    Returns:
    -------
        -(1/x^2) * y

    """
    return -math.pow(inv(x), 2) * y


def relu_back(x: float, y: float) -> float:
    """Compute the derivative of ReLU with respect to x, scaled by y.

    This is the backward pass for the ReLU function.

    Args:
    ----
        x: Input to the original ReLU function
        y: Gradient from the downstream computation

    Returns:
    -------
        y if x > 0, otherwise 0

    """
    return y if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], it: Iterable[float]) -> List[float]:
    """Apply a function to each element of an iterable.

    Args:
    ----
        f: Function to apply to each element
        it: Iterable of numbers

    Returns:
    -------
        List with f applied to each element

    """
    return [f(x) for x in it]


def zipWith(
    f: Callable[[float, float], float], it1: Iterable[float], it2: Iterable[float]
) -> List[float]:
    """Apply a binary function to corresponding elements of two iterables.

    Args:
    ----
        f: Binary function to apply
        it1: First iterable
        it2: Second iterable

    Returns:
    -------
        List with f applied to corresponding pairs

    """
    return [f(x, y) for x, y in zip(it1, it2)]


def reduce(f: Callable[[float, float], float], it: Iterable[float]) -> float | None:
    """Reduce an iterable to a single value using a binary function.

    Args:
    ----
        f: Binary function to use for reduction
        it: Iterable to reduce

    Returns:
    -------
        Single reduced value, or None if iterable is empty

    """
    iterator = iter(it)
    r1 = next(iterator, None)
    if r1 is None:
        return None

    r2 = next(iterator, None)
    while r2 is not None:
        r1 = f(r1, r2)
        r2 = next(iterator, None)

    return r1


def negList(x: List[float]) -> List[float]:
    """Negate all elements in a list.

    Args:
    ----
        x: List of numbers

    Returns:
    -------
        List with all elements negated

    """
    return map(neg, x)


def addLists(x: List[float], y: List[float]) -> List[float]:
    """Add corresponding elements of two lists.

    Args:
    ----
        x: First list
        y: Second list

    Returns:
    -------
        List with element-wise sums

    """
    return zipWith(add, x, y)


def sum(x: List[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
        x: List of numbers

    Returns:
    -------
        Sum of all elements, or 0 if list is empty

    """
    r = reduce(add, x)
    return r if r is not None else 0


def prod(x: List[float]) -> float:
    """Compute the product of all elements in a list.

    Args:
    ----
        x: List of numbers

    Returns:
    -------
        Product of all elements, or 1 if list is empty

    """
    r = reduce(mul, x)
    return r if r is not None else 1
