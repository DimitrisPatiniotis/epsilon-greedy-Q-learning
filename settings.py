from typing import List

epsilon: float = 1
decay: float = 0.01
learning_rate: float = 0.05
gamma: float = 0.9

capacity: List[List[float]] = [
    [2.5, 2.5],
    [2.5, 2.5]
]

actions: List[List[int]] = [
    [0, 0], [0, 1],
    [1, 0], [1, 1]
]