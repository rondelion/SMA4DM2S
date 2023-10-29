# RL Agent for a minimal delayed Match-to-Sample task

import sys
import json
import gym

from tensorforce.environments import Environment
from tensorforce.agents import Agent

def train(n, agent, environment):
    for _ in range(n):
        states = environment.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)


def evaluate(n, agent, environment):
    sum_rewards = 0.0
    for _ in range(n):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
    return sum_rewards / n


def main():
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)

    # Create agent and environment
    env = gym.make(config['env']['name'], config=config['env'])
    environment = Environment.create(environment=env, max_episode_timesteps=15)
    agent_type = sys.argv[2]
    if agent_type == "dqn":
        agent = Agent.create(agent=agent_type, environment=environment, batch_size=10, memory=100)
    else:
        agent = Agent.create(agent=agent_type, environment=environment, batch_size=10)

    for _ in range(100):
        train(100, agent, environment)
        avr_rewards = evaluate(100, agent, environment)
        print('Mean episode reward:', avr_rewards)

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()
