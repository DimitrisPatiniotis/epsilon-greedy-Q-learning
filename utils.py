from typing import List, Tuple, Union
import agent

def generate_agents(number_weight_tuple_list: List[Tuple[int, float]]) -> Tuple[List[agent.Agent], int]:
    """Generate a list of agents based on the given number-weight tuple list."""
    agent_list: List[agent.Agent] = []
    
    for agent_num, (number, weight) in enumerate(number_weight_tuple_list, start=1):
        agent_list.extend([agent.Agent(f'Agent {agent_num}', weight) for _ in range(number)])
        
    return agent_list, agent_num

def round_list_numbers(alist: List[float]) -> List[float]:
    """Round the numbers in the list to one decimal place."""
    return [round(f, 1) for f in alist]

def calculate_total_distribution(agents: List[agent.Agent]) -> List[List[float]]:
    """Calculate the total distribution of actions chosen by all agents."""
    distribution: List[List[float]] = [[0, 0], [0, 0]]
    for i in agents:
        distribution[i.move[0]][i.move[1]] += round(i.weight, 1)
    return [[round(f, 1) for f in i] for i in distribution]

def get_reward(state: List[List[float]]) -> List[List[float]]:
    """Calculate the reward based on the current state."""
    return [[round(2.5 - value, 1) for value in row] for row in state]

def distribute_reward(agents: List[agent.Agent], reward: List[List[float]]) -> None:
    """Distribute the reward to the agents based on their chosen actions."""
    for i in agents:
        i.reward = reward[i.move[0]][i.move[1]]
        i.accumulated_reward += reward[i.move[0]][i.move[1]]


def update_q_values(agents: List[agent.Agent]) -> None:
    """Update the Q-values for all agents."""
    for i in agents:
        i.update_q()

def test_overweight(state: List[List[float]]) -> Union[List[int], bool]:
    """Check if the state is overweight and return the action causing the overweight condition."""
    if state[0][0] > 2.5:
        return [0, 0]
    elif state[0][1] > 2.5:
        return [0, 1]
    elif state[1][0] > 2.5:
        return [1, 0]
    elif state[1][1] > 2.5:
        return [1, 1]
    else:
        return False

def get_avg_total_r(agents: List[agent.Agent]) -> float:
    """Calculate the average total accumulated reward of all agents."""
    return sum([i.accumulated_reward for i in agents])

def get_avg_r_by_weight(weight: float, agents: List[agent.Agent]) -> float:
    """Calculate the average reward of agents with a specific weight."""
    agent_list: List[agent.Agent] = [i for i in agents if i.weight == weight]
    return sum([i.reward for i in agent_list]) / len(agent_list)

if __name__ == '__main__':
    print('utils module')