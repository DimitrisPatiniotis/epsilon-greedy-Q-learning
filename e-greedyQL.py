from typing import List
import argparse
import plots
import utils

def e_greedyQL(iter: int, best_train: bool) -> None:

    agents, len_agents = utils.generate_agents([(2, 0.6), (5, 0.4), (8, 0.2)])
    total_mistakes: int = 0
    mistakes_per_turn: List[float] = []
    avg_rewards: List[float] = []

    six_tr: List[float] = []
    four_tr: List[float] = []
    two_tr: List[float] = [] 
    pos_1: List[float] = []
    pos_2: List[float] = []
    pos_3: List[float] = []
    pos_4: List[float] = []

    for _ in range(1, iter):
        # Each agent makes a move
        for i in agents:
            i.make_move()
        # Calculate the current state of the game
        turn_state = utils.calculate_total_distribution(agents)

        # Continue the loop while the state is overweight
        while utils.test_overweight(turn_state) != False:
            # Increment the total mistakes counter
            total_mistakes += 1
            # Check each agent's move
            for i in agents:
                if i.move == utils.test_overweight(turn_state):
                    if best_train:
                        # Get rewards for the current state
                        r = utils.get_reward(turn_state)
                        # Distribute rewards to the agent
                        utils.distribute_reward([i], r)
                        # Update Q-values for the agent
                        utils.update_q_values([i])
                        # Append rewards to the corresponding lists
                        pos_1.append(r[0][0])
                        pos_2.append(r[0][1])
                        pos_3.append(r[1][0])
                        pos_4.append(r[1][1])
                    # Make a move for the agent
                    i.make_move()
                    # Break the loop once the agent's move is found
                    break

            # Recalculate the current state of the game
            turn_state = utils.calculate_total_distribution(agents)

        # Get rewards for the current state
        r = utils.get_reward(turn_state)
        # Distribute rewards to all agents
        utils.distribute_reward(agents, r)
        # Update Q-values for all agents
        utils.update_q_values(agents)
        # Append rewards for the first, second, and third tiles to the respective lists
        six_tr.append(r[0][0])
        four_tr.append(r[0][1])
        two_tr.append(r[1][0])

        # Reduce epsilon value for each agent
        for i in agents:
            i.reduce_epsilon()

        # Append the total mistakes to the list
        mistakes_per_turn.append(total_mistakes)
        # Reset the total mistakes counter
        total_mistakes = 0
        # Calculate and append the average accumulated rewards for all agents
        avg_rewards.append(utils.get_avg_total_r(agents))

    plots.plot_mistakes_per_episode(mistakes_per_turn)
    plots.plot_episode_rewards(avg_rewards)
    plots.plot_agent_rewards(six_tr, 0.6)
    plots.plot_agent_rewards(four_tr, 0.4)
    plots.plot_agent_rewards(two_tr, 0.2)
    plots.plot_reward_per_tile(pos_1, pos_2, pos_3, pos_4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the main function with a specified number of iterations.')
    parser.add_argument('--iterations', type=int, help='Number of iterations.')
    parser.add_argument('--best_train', action='store_true', help='Flag to enable best training mode.')

    args = parser.parse_args()
    iterations = args.iterations
    best_train = args.best_train

    e_greedyQL(iterations, best_train)