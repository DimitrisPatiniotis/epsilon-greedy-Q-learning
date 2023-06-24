from typing import Union, List, Tuple
import numpy as np
import random
import settings

class Agent:
    
    def __init__(self, name: str, weight: float):
        self.name: str = name
        self.weight: float = weight
        self.epsilon: float = settings.epsilon
        self.q_table: List[float] = [0, 0, 0, 0]
        self.actions: List[List[int]] = settings.actions
        self.decay: float = settings.decay
        self.move: List[int] = None
        self.reward: float = None
        self.accumulated_reward: float = 0
        self.gamma: float = settings.gamma
        self.learning_rate: float = settings.learning_rate

    def reduce_epsilon(self) -> None:
        """Reduce the epsilon value for exploration over time."""
        if self.epsilon != 0.:
            self.epsilon = round((self.epsilon - self.decay), 2)

    def select_move(self) -> int:
        """Select an action based on epsilon-greedy policy."""
        if self.epsilon <= 0:
            return np.argmax(self.q_table)
        elif np.random.rand() <= self.epsilon:
            return random.randrange(0, len(self.q_table))
        else:
            return np.argmax(self.q_table)
    
    def make_move(self) -> List[int]:
        """Make a move by selecting an action."""
        self.move = self.actions[self.select_move()]
        self.move_num = self.actions.index(self.move)
        return self.move
    
    def get_reward(self, reward: float) -> None:
        """Set the reward received by the agent."""
        self.reward = reward
        self.accumulated_reward += reward


    def update_q(self) -> None:
        """Update the Q-value based on the Q-learning algorithm."""
        index = self.actions.index(self.move)
        self.q_table[index] = self.q_table[index] + self.learning_rate * (self.reward + (self.gamma * np.max(self.q_table)) - self.q_table[index])

if __name__ == '__main__':
    print('agent module')