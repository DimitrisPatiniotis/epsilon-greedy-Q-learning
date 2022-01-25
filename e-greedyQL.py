import numpy as np
import random

epsilon = 1
decay = 0.01

capacity = [
    [2.5, 2.5],
    [2.5, 2.5]
]

actions = [
    [0,0] , [0,1],
    [1,0] , [1,1]
]

Q_table = [0, 0, 0, 0]

class Agent:
    
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        self.epsilon = epsilon
        self.q_table = Q_table
        self.actions = actions
        self.move = None
        self.reward = None
    
    def select_move(self):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0, len(self.q_table))
        else:
            return np.argmax(self.q_table)
    
    def make_move(self):
        self.move = actions[self.select_move()]
        self.move_num = self.actions.index(self.move)
        return self.move
    
    def get_reward(self, reward):
        self.reward = reward

    def update_q(self):
        pass

def generate_agents(number_weight_tuple_list):
    agent_num = 1
    agent_list = []
    for i in number_weight_tuple_list:
        for x in range(i[0]):
            agent_list.append(Agent('Agent {}'.format(agent_num), i[1]))
            agent_num += 1
    agent_num -= 1
    return agent_list, agent_num

def round_list_numbers(alist):
    return [round(f, 1) for f in alist]

def calculate_total_distribution(agents):
    distribution = [[0,0], [0,0]]
    for i in [i for i in agents]: distribution[i.move[0]][i.move[1]] += round(i.weight,1)
    rounded_distr = []
    for i in distribution: rounded_distr.append(round_list_numbers(i))
    return rounded_distr

def distribute_reward(agents, state):
    # Initialize Reward List
    reward = [[0,0],[0,0]]
    for i in range(len(state)):
        for x in range(len(state[i])):
            reward[i][x] = round((2.5 - state[i][x]),1)
    for i in agents:
        i.reward = reward[i.move[0]][i.move[1]]

def update_q_values(agents):
    for i in agents: i.update_q()


def main():
    agents, len_agents = generate_agents([(2,0.6), (5, 0.4), (8, 0.2)])
    agent_weights = [f.weight for f in agents]
    for i in agents: i.make_move()
    turn_state = calculate_total_distribution(agents)
    distribute_reward(agents, turn_state)
    update_q_values(agents)

if __name__ == '__main__':
    main()