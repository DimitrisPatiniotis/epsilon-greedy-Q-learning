# Epsilon-Greedy Q-Learning in a Multi-agent Environment

## Table of contents
* [General Overview and Goals](#general-overview-and-goals)
* [Problem Description](#problem-description)
* [Solution Overview](#solution-overview)
* [Usage](#usage)

## General Overview and Goals

This repository shows **how to implement the Epsilon Greedy Q-learning algorithm in a multi-agent environment**. The agents are trained in a cooperative setting to maximize their total reward. The goal of this repository is to show a **simple implementation** of the Epsilon-Greedy algorithm, with **step by step annotations and explanations** for every main functionality.

## Problem Description

In a **2x2 grid**, each tile has a weight **capacity limit of 2.5 units**. Agents with different weights move within the grid and need to **coordinate their actions to prevent any tile from exceeding its weight threshold**. If an overweight condition occurs, the agents must readjust their moves and learn to coordinate better to achieve a balanced distribution of weight across the tiles.

## Solution Overview

The solution involves training the agents through iterations, using the Epsilon-Greedy Q-Learning algorithm, allowing them to learn optimal strategies for weight coordination. Here is an overview of the **key components and processes** in the code:

1. **Agent Generation:** The code generates a list of agents based on the specified number-weight tuples. Each agent is assigned a unique identifier and weight value.

2. **Agent Moves:** In each iteration, the agents make moves based on their weights. These moves determine the distribution of weight across the grid tiles.

3. **Overweight Condition Handling:** After the agents make their moves, the code checks if any tile exceeds its weight threshold. If an overweight condition is detected, the agents are penalized and prompted to readjust their moves.

4. **Rewards and Q-Value Updates:** Rewards are calculated based on the current state of the grid, and they are distributed to the agents. Q-values for the current state are updated to improve the agents' decision-making process.

5. **Exploration-Exploitation Trade-off:** The agents' exploration behavior is controlled by adjusting the epsilon value. Over time, the epsilon value is reduced to favor exploitation of learned strategies.

6. **Metrics Tracking:** Various metrics are tracked throughout the training process, including mistakes per turn, average accumulated rewards, and average rewards based on agent weights. These metrics provide insights into the agents' performance and the progress of training.

## Usage

1. Clone repo

```
$ git clone git@github.com:DimitrisPatiniotis/epsilon-greedy-Q-learning.git
```

2. Create a virtual environment and install all requirements listed in requirements.txt

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

3. Run Algorithm

```
$ python e-greedyQL.py --iterations <num_iterations> --best_train
```

Replace <num_iterations> with the desired number of iterations for training.