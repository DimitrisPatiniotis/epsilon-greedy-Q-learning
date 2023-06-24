import matplotlib.pyplot as plt
from typing import List

def plot_episode_rewards(avg_rewards: List[float]) -> None:
    """
    Plot the average accumulated rewards per episode.
    """
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Accumulated Reward')
    plt.title('Average Accumulated Rewards Per Episode')
    plt.grid(True)
    plt.show()

def plot_agent_rewards(agent_rewards: List[float], weight: float) -> None:
    """
    Plot the average episode rewards per agent.
    """
    plt.plot(range(1, len(agent_rewards) + 1), agent_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Average Episode Reward Per Agent ({weight} weight)')
    plt.grid(True)
    plt.show()

def plot_mistakes_per_episode(mistakes_per_turn: List[int]) -> None:
    """
    Plot the average mistakes per episode.
    """
    plt.plot(range(1, len(mistakes_per_turn) + 1), mistakes_per_turn)
    plt.xlabel('Episodes')
    plt.ylabel('Mistakes')
    plt.title('Average Mistakes Per Episode')
    plt.grid(True)
    plt.show()

def plot_reward_per_tile(pos_1: List[float], pos_2: List[float], pos_3: List[float], pos_4: List[float]) -> None:
    """
    Plot the rewards of each tile per turn.
    """
    turns = range(len(pos_1))
    plt.plot(turns, pos_1, label='Reward of 1st tile')
    plt.plot(turns, pos_2, label='Reward of 2nd tile')
    plt.plot(turns, pos_3, label='Reward of 3rd tile')
    plt.plot(turns, pos_4, label='Reward of 4th tile')
    plt.xlabel('Turns')
    plt.ylabel('Reward')
    plt.title('Reward of tile per turn')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print('plotting utils')