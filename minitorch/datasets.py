"""Dataset generation functions for minitorch classification tasks.

This module provides various 2D dataset generators for testing classification algorithms.
Each function generates points in the unit square [0,1] x [0,1] with binary labels.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points in the unit square [0,1] x [0,1].

    Args:
    ----
        N: Number of points to generate

    Returns:
    -------
        List of (x, y) coordinate tuples

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Container for a 2D classification dataset.

    Attributes
    ----------
        N: Number of data points
        X: List of (x, y) coordinate tuples
        y: List of binary labels (0 or 1)

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple linear classification dataset.

    Points are labeled 1 if x < 0.5, otherwise 0.

    Args:
    ----
        N: Number of points to generate

    Returns:
    -------
        Graph object containing points and labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a diagonal classification dataset.

    Points are labeled 1 if x + y < 0.5, otherwise 0.

    Args:
    ----
        N: Number of points to generate

    Returns:
    -------
        Graph object containing points and labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a split classification dataset.

    Points are labeled 1 if x < 0.2 or x > 0.8, otherwise 0.

    Args:
    ----
        N: Number of points to generate

    Returns:
    -------
        Graph object containing points and labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate an XOR classification dataset.

    Points are labeled 1 if they are in opposite quadrants (top-left or bottom-right), otherwise 0.

    Args:
    ----
        N: Number of points to generate

    Returns:
    -------
        Graph object containing points and labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a circular classification dataset.

    Points are labeled 1 if they are outside a circle centered at (0.5, 0.5) with radius sqrt(0.1), otherwise 0.

    Args:
    ----
        N: Number of points to generate

    Returns:
    -------
        Graph object containing points and labels

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a spiral classification dataset.

    Creates two interleaved spirals with different labels.

    Args:
    ----
        N: Number of points to generate

    Returns:
    -------
        Graph object containing points and labels

    """

    def x(t: float) -> float:
        """Calculate x coordinate for spiral at parameter t."""
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Calculate y coordinate for spiral at parameter t."""
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
