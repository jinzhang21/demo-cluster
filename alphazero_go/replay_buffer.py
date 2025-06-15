import random
from collections import deque
from typing import List, Tuple

import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
