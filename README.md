# Taxi-V3 RL Algorithms

This repository contains implementations of several reinforcement learning (RL) algorithms to solve the Taxi-v3 environment from the OpenAI Gym. The algorithms include Value Iteration, Policy Iteration, Q-Learning, and Direct Evaluation (Monte Carlo). The project also includes functionality to visualize the performance of the trained policies through video recordings.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
  - [Value Iteration](#value-iteration)
  - [Policy Iteration](#policy-iteration)
  - [Q-Learning](#q-learning)
  - [Direct Evaluation (Monte Carlo)](#direct-evaluation-monte-carlo)
- [Testing](#testing)
- [Credits](#credits)

## Installation

To run the code, you need to have Python installed along with the required libraries. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To test a policy and visualize it, use the `test_policy_video` function. For instance:

```python
from IPython.display import HTML

# Assuming `optimal_policy` is defined
video_html = test_policy_video(optimal_policy, 'policy-name')
HTML(video_html)
```

## Algorithms

### Value Iteration

1. **Initializing Parameters**

    ```python
    taxi = Taxi(num_iter=200, num_evaluate_steps=20, discount_rate=0.8)
    ```

2. **Finding Optimal State Values**

    ```python
    def value_iteration(vtable, num_iter, discount_rate):
        for _ in range(num_iter):
            v_old = np.copy(vtable)
            for state in range(taxi.state_size):
                temp_action = np.zeros(taxi.action_size)
                for action in range(taxi.action_size):
                    action_expected_value = 0
                    transitions = taxi.env.P[state][action]
                    if type(transitions) == tuple:
                        transitions = [transitions]
                    for transition in transitions:
                        action_expected_value += transition[0] * (transition[2] + discount_rate * vtable[transition[1]])
                    temp_action[action] = action_expected_value
                vtable[state] = np.max(temp_action)
    ```

3. **Extracting The Optimal Policy**

    ```python
    def optimal_policy_extraction(vtable, ptable, num_iter, discount_rate):
        value_iteration(vtable, num_iter, discount_rate)
        for state in range(taxi.state_size):
            temp_action = np.zeros(taxi.action_size)
            for action in range(taxi.action_size):
                action_expected_value = 0
                transitions = taxi.env.P[state][action]
                if type(transitions) == tuple:
                    transitions = [transitions]
                for transition in transitions:
                    action_expected_value += transition[0] * (transition[2] + discount_rate * vtable[transition[1]])
                temp_action[action] = action_expected_value
            ptable[state] = np.argmax(temp_action)
    ```

4. **Running The Algorithm**

    ```python
    optimal_policy_extraction(taxi.vtable, taxi.ptable, taxi.num_iter, taxi.discount_rate)
    optimal_policy = taxi.ptable.copy()
    ```

### Policy Iteration

1. **Initialize Parameters**

    ```python
    taxi = Taxi(num_iter=200, num_evaluate_steps=20, discount_rate=0.8)
    ```

2. **Policy Evaluation**

    ```python
    def evaluate(num_iter, discount_rate):
        vtable = np.zeros(taxi.state_size)
        for _ in range(num_iter):
            v_old = np.copy(vtable)
            for state in range(taxi.state_size):
                action = taxi.ptable[state]
                action_expected_value = 0
                transitions = taxi.env.P[state][action]
                if type(transitions) == tuple:
                    transitions = [transitions]
                for transition in transitions:
                    action_expected_value += transition[0] * (transition[2] + discount_rate * vtable[transition[1]])
                vtable[state] = action_expected_value
        return vtable
    ```

3. **Policy Improvement**

    ```python
    def improvement(ptable, num_iter, num_evaluate_steps, discount_rate):
        for _ in range(num_iter):
            vtable = evaluate(num_evaluate_steps, discount_rate).copy()
            for state in range(taxi.state_size):
                temp_action = np.zeros(taxi.action_size)
                for action in range(taxi.action_size):
                    action_expected_value = 0
                    transitions = taxi.env.P[state][action]
                    if type(transitions) == tuple:
                        transitions = [transitions]
                    for transition in transitions:
                        action_expected_value += transition[0] * (transition[2] + discount_rate * vtable[transition[1]])
                    temp_action[action] = action_expected_value
                ptable[state] = np.argmax(temp_action)
    ```

4. **Running The Algorithm**

    ```python
    improvement(taxi.ptable, taxi.num_iter, taxi.num_evaluate_steps, taxi.discount_rate)
    ```

### Q-Learning

1. **Initializing Parameters**

    ```python
    taxi = Taxi(num_episodes=10000, max_steps=99, learning_rate=0.9, discount_rate=0.8, epsilon=1.0, decay_rate=0.005)
    ```

2. **Training**

    ```python
    def q_learning_train(qtable, num_episodes, max_steps, learning_rate, discount_rate, epsilon, decay_rate):
        for episode in range(num_episodes):
            state, info = taxi.env.reset()
            done = False
            for s in range(max_steps):
                if random.uniform(0, 1) < epsilon:
                    action = taxi.env.action_space.sample()
                else:
                    action = np.argmax(qtable[state])
                new_state, reward, done, truncated, info = taxi.env.step(action)
                qtable[state][action] = (1 - learning_rate) * qtable[state][action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state]))
                state = new_state
                if done:
                    break
            epsilon = np.exp(-decay_rate * episode)
    ```

3. **Running The Algorithm**

    ```python
    q_learning_train(taxi.qtable, taxi.num_episodes, taxi.max_steps, taxi.learning_rate, taxi.discount_rate, taxi.epsilon, taxi.decay_rate)
    ```

4. **Extracting The Policy**

    ```python
    for state in range(taxi.state_size):
        taxi.ptable[state] = np.argmax(taxi.qtable[state][:])
    ```

### Direct Evaluation (Monte Carlo)

1. **Initializing Parameters**

    ```python
    taxi = Taxi(num_episodes=10000, max_steps=99, discount_rate=0.8)
    ```

2. **Training**

    ```python
    def monte_carlo(ptable, num_episodes, max_steps, discount_rate):
        count = np.ones(taxi.state_size)
        vtable = np.zeros(taxi.state_size)
        for _ in range(num_episodes):
            state, info = taxi.env.reset()
            done = False
            trajectory = [state]
            rewards = []
            for s in range(max_steps):
                new_state, reward, done, truncated, info = taxi.env.step(ptable[state])
                trajectory.append(new_state)
                count[new_state] += 1
                rewards.append(reward)
                if done:
                    break
            discounted_reward_sum = 0
            rewards = list(reversed(rewards))
            trajectory = list(reversed(trajectory))[1:]
            for i in range(len(trajectory)):
                discounted_reward_sum += rewards[i]
                vtable[trajectory[i]] += discounted_reward_sum
                discounted_reward_sum *= discount_rate
        vtable /= count
        return vtable
    ```

3. **Testing**

    ```python
    optimal_policy_state_values = monte_carlo(optimal_policy, taxi.num_episodes, taxi.max_steps, taxi.discount_rate)
    print(optimal_policy_state_values)
    ```

## Testing

Each algorithm's performance can be tested and visualized using the `test_policy_video` function. This function records the environment's behavior under the specified policy and returns an HTML video element for display in Jupyter Notebooks.

## Credits

This project uses the Taxi-v3 environment from OpenAI's Gym. The code structure and implementation details are inspired by common RL practices and algorithms.
