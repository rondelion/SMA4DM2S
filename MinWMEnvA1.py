import gym
import numpy as np


class MinWMEnvA1(gym.Env):
    def __init__(self, config):
        self.action_space = gym.spaces.Discrete(4)
        self.attention_size = config["attention_size"]
        self.atttibute_number = config["attribute_number"]
        self.obs_dim = self.attention_size * (self.atttibute_number + 2)
        self.observation_space = gym.spaces.Box(low=np.zeros(self.obs_dim, dtype='int'),
                                                high=np.ones(self.obs_dim, dtype='int'))
        self.task_switch = 0
        self.sample = np.zeros(self.atttibute_number, dtype='int')
        self.target = np.zeros(self.atttibute_number, dtype='int')
        self.sample_array = np.array([], dtype='int')
        self.target_array = np.array([], dtype='int')
        self.control = np.array([0, 0])
        self.switch_period = config["switch_period"]
        self.sample_period = config["sample_period"]
        self.response_period = config["response_period"]
        self.match_delay = config["match_delay"]
        self.reward_delay = config["reward_delay"]
        self.action = 0
        self.done = False
        self.count = 0

    def reset(self):
        self.action = 0
        self.done = False
        self.count = 0
        self.task_switch = np.random.randint(0, self.atttibute_number, dtype='int')
        self.sample = np.random.randint(0, self.attention_size, dtype='int')
        self.target = np.random.randint(0, self.attention_size, dtype='int')
        return np.zeros(self.obs_dim, dtype='int')

    def one_hot(self, n, dim):
        buf = np.array([], dtype='int')
        for i in range(dim):
            if i == n:
                buf = np.append(buf, np.array([1], dtype='int'))
            else:
                buf = np.append(buf, np.array([0], dtype='int'))
        return buf

    def step(self, action):
        reward = 0
        self.count += 1
        observation = np.zeros(self.obs_dim, dtype='int')
        task_switch = np.zeros(self.attention_size, dtype='int')
        if self.count <= self.switch_period: # + 1:
            # showing the task switch
            for i in range(self.atttibute_number):
                if i == self.task_switch:
                    observation[i] = 1
        elif self.count <= self.switch_period + self.match_delay: # + 1:  # match delay
            # prepare for the showing period
            self.sample_array = np.array([], dtype='int')
            for i in range(self.atttibute_number):
                if i == self.task_switch:
                    n = self.sample
                else:
                    n = np.random.randint(0, self.attention_size, dtype='int')
                self.sample_array = np.append(self.sample_array, self.one_hot(n, self.attention_size))
        elif self.count <= self.switch_period + self.match_delay + self.sample_period: # + 1:
            # showing the sample
            control = np.zeros(self.attention_size, dtype='int')
            control[0] = 1  # [1, 0, ...]
            observation = np.append(task_switch, np.append(self.sample_array, control))
        elif self.count <= self.switch_period + self.sample_period + self.match_delay * 2: # + 1:   # match delay
            # prepare for the response period
            self.target_array = np.array([], dtype='int')
            for i in range(self.atttibute_number):
                if i == self.task_switch:
                    n = self.target
                else:
                    n = np.random.randint(0, self.attention_size, dtype='int')
                self.target_array = np.append(self.target_array, self.one_hot(n, self.attention_size))
        elif self.count <= self.switch_period + self.sample_period + self.response_period + self.match_delay * 2: # + 1:
            # response period
            control = np.zeros(self.attention_size, dtype='int')
            control[0] = 0
            control[1] = 1  # [1, 1, ...]
            observation = np.append(task_switch, np.append(self.target_array, control))
            if action > 0 and self.count > self.sample_period + self.match_delay * 2 + 1: # + 2:
                self.action = action    # response
        elif self.count <= self.switch_period + self.sample_period + self.match_delay * 2 + self.response_period + self.reward_delay: # + 1:
            if action > 0 and self.count <= self.switch_period + self.sample_period + self.match_delay * 2 + self.response_period + 1:
                self.action = action    # response
        else:
            reward = 0.0
            if self.sample == self.target and self.action == 2:
                reward = 1.0
            if self.sample != self.target and self.action == 1:
                reward = 1.0
            self.done = True
        return observation, reward, self.done, {}

    def render(self):
        pass


def main():
    config = {"attribute_number": 3, "attention_size": 3, "switch_period": 2, "sample_period": 3, "response_period": 3,
              "match_delay": 1, "reward_delay": 1}
    env = MinWMEnvA1(config)
    for i in range(10):
        obs = env.reset()
        action = np.random.randint(1, 3)
        while True:
            control = obs[env.attention_size * (env.atttibute_number + 1):][0:2]
            # if np.array_equal(control, np.array([1, 1])):
            #     action = np.random.randint(0, 4)
            # else:
            #     action = 0
            print(obs, action)  # , end=",")
            obs, reward, done, info = env.step(action)
            if done:
                print("reward:", reward)
                break


if __name__ == '__main__':
    main()
