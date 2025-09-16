"""Minitorch: A minimal educational PyTorch-like neural network library.

This package provides core functionality for building and training neural networks,
including mathematical operators, modules, datasets, and testing utilities.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
