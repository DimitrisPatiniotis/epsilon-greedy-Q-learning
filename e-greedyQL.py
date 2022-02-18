import numpy as np
import random
import matplotlib.pyplot as plt

epsilon = 1
decay = 0.01
learning_rate = 0.05
gamma = 0.9

capacity = [
    [2.5, 2.5],
    [2.5, 2.5]
]

actions = [
    [0,0] , [0,1],
    [1,0] , [1,1]
]


class Agent:
    
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        self.epsilon = epsilon
        self.q_table = [0, 0, 0, 0]
        self.actions = actions
        self.decay = decay
        self.move = None
        self.reward = None
        self.total_reward = 0

    def reduce_epsilon(self):
        if self.epsilon != 0.:
            self.epsilon = round((self.epsilon - self.decay), 2)

    def select_move(self):
        if self.epsilon <= 0:
            return np.argmax(self.q_table)
        elif np.random.rand() <= self.epsilon:
            return random.randrange(0, len(self.q_table))
        else:
            return np.argmax(self.q_table)
    
    def make_move(self):
        self.move = actions[self.select_move()]
        self.move_num = self.actions.index(self.move)
        return self.move
    
    def get_reward(self, reward):
        self.reward = reward
        self.total_reward += reward
        print(self.total_reward)

    def update_q(self):
        index = self.actions.index(self.move)
        self.q_table[index] = self.q_table[index] + learning_rate * (self.reward + (gamma * np.max(self.q_table)) - self.q_table[index])



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
        i.total_reward += reward[i.move[0]][i.move[1]]


def update_q_values(agents):
    for i in agents: i.update_q()

def test_overweight(state):
    if state[0][0] > 2.5:
        return [0, 0]
    elif state[0][1] > 2.5:
        return [0,1]
    elif state[1][0] > 2.5:
        return [1,0]
    elif state[1][1] > 2.5:
        return [1,1]
    else:
        return False

def get_avg_total_r(agents):
    return sum([i.total_reward for i in agents])

def get_avg_r_by_weight(weight, agents):
    agent_list = [i for i in agents if i.weight == weight]
    return sum([i.reward for i in agent_list])/len(agent_list)


def main(iter, best_train):
    its = iter
    agents, len_agents = generate_agents([(2,0.6), (5, 0.4), (8, 0.2)])
    total_mistakes = 0
    mistakes_per_turn, avg_rewards = [], []

    six_tr, four_tr, two_tr = [], [], [] 

    for turn in range(1,its):
        for i in agents: i.make_move()
        turn_state = calculate_total_distribution(agents)
        
        time = 0

        while test_overweight(turn_state) != False:
            total_mistakes += 1
            for i in agents:
                if i.move == test_overweight(turn_state):
                    if best_train:
                        distribute_reward([i], turn_state)
                        update_q_values([i])
                    i.make_move()
            turn_state = calculate_total_distribution(agents)


        distribute_reward(agents, turn_state)
        update_q_values(agents)
        if turn%20 == 0:
            for i in agents:
                i.reduce_epsilon()

        mistakes_per_turn.append(round(total_mistakes/turn, 3))
        avg_rewards.append(get_avg_total_r(agents))

        six_tr.append(get_avg_r_by_weight(0.6, agents))
        four_tr.append(get_avg_r_by_weight(0.4, agents))
        two_tr.append(get_avg_r_by_weight(0.2, agents))

    fig, ax= plt.subplots()
    ax.plot(range(1,its), mistakes_per_turn)
    ax.set(xlabel='episodes', ylabel='mistakes', title='Avarage Mistakes Per Episode')
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(range(1,its), avg_rewards)
    ax.set(xlabel='episodes', ylabel='avg total rewards', title='Avarage Total Rewards Per Episode')
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(range(1,its), six_tr)
    ax.set(xlabel='episodes', ylabel='reward', title='Avarage Episode Reward Per Agent (0.6 weight)')
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(range(1,its), four_tr)
    ax.set(xlabel='episodes', ylabel='reward', title='Avarage Episode Reward Per Agent (0.4 weight)')
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(range(1,its), two_tr)
    ax.set(xlabel='episodes', ylabel='reward', title='Avarage Episode Reward Per Agent (0.2 weight)')
    ax.grid()
    plt.show()



if __name__ == '__main__':
    main(2500, best_train=False)