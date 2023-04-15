# A Sequence Memory Agent for a Delayed Match-to-Sample Task

import gym
import numpy as np

import sys
import os
import argparse
import json

import brica1.brica_gym
import brical


class EpisodicMemory(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.attention_size = config['env']["attention_size"]
        self.atttibute_number = config['env']["attribute_number"]
        self.obs_dim = self.attention_size * (self.atttibute_number + 2)
        self.obs_span = self.atttibute_number + 2   # + task_switch:1 + control:1
        self.action_dim = config['env']['actions']
        self.min_success_rate = config['agent']['min_success_rate']
        self.actors = config['agent']['actors']
        self.make_in_port('gated_obs', self.obs_dim)
        self.make_in_port('action', 4)
        self.make_in_port('obs_changed', 1)
        self.make_in_port('token_in', 1)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('attention', self.obs_span)
        self.make_out_port('action', self.action_dim)
        self.make_out_port('token_out', 1)
        self.init_choices = {'attention': None, 'action': None}
        self.act_vector_dim = self.obs_span + 2 + self.action_dim
        self.actor_chooser = Chooser(3)
        self.episode = []
        self.hypotheses = []
        self.prev_node = None
        self.prev_observation = None
        self.path_indices = {}
        self.on_hypothesis = False
        self.prev_obs_changed = 1
        self.prev_recog = 0
        self.prev_prev_actor = ""
        self.prev_prev_choice = None
        self.reward = 0
        self._env = config["env"]   # for debug
        self._task_switch = []  # for debug
        self._feature = []      # for debug
        self._correct = 0       # for debug
        self._correct_response = 0

    def fire(self):
        if self.get_in_port('done').buffer[0] == 1:
            # if self.get_in_port('reward').buffer[0] > 0:
            #     print(" reward:" + str(self.get_in_port('reward').buffer[0]))
            self.update_hypotheses(self.get_in_port('reward').buffer[0])
            return
        attention = [-1] * self.obs_span  # no choice
        action = [-1] * self.action_dim  # no choice
        gated_obs = self.in_ports['gated_obs'].buffer
        obs_changed = self.get_in_port('obs_changed').buffer[0]
        recog = self.recog(gated_obs, obs_changed)
        observation = np.concatenate([gated_obs, np.array([recog])])
        node = None
        act = None
        prev_actor = None
        prev_choice = None
        prev_action = {"action": self.get_in_port('action').buffer,
                       "attention": self.get_in_port('attention').buffer}
        if self.prev_observation is not None:
            prev_actor = self.get_executed_actor(prev_action)
            if prev_actor is not None:
                prev_choice = prev_action[prev_actor]
            if max(self.prev_observation) > 0:
                if prev_actor == "action":
                    self.episode.append((self.prev_observation, "action", prev_choice, self.prev_recog))
                elif self.prev_prev_actor == "attention":
                    self.episode.append((self.prev_observation, "attention", self.prev_prev_choice, self.prev_recog))
        self.correct_path(gated_obs, prev_actor, prev_choice)
        if self.on_hypothesis:
            if max(observation) > 0:
                if max(self.prev_observation) == 0:
                    if self.prev_node.parent.actor == prev_actor and tuple(self.prev_node.parent.choice) == tuple(prev_choice) \
                        and self.prev_node.parent is not None and self.prev_node.parent.observation == tuple(observation):
                        if prev_actor is not None:
                            node = self.prev_node.parent
                        else:
                            node = self.prev_node   # continue
                else:
                    if self.prev_node.actor == prev_actor and tuple(self.prev_node.choice) == tuple(prev_choice) \
                        and self.prev_node.parent is not None and self.prev_node.parent.observation == tuple(observation):
                        if prev_actor is not None:
                            node = self.prev_node.parent
                        else:
                            node = self.prev_node   # continue
            else:
                node = self.prev_node  # continue
        if node is None:
            self.on_hypothesis = False
            node = self.find_node(observation)
            if node is not None:
                self.on_hypothesis = True
        if node is not None:
            if max(observation) == 0 and node.parent is not None and node.parent.actor == "attention":
                actor = node.parent.actor
                choice = node.parent.choice
            else:
                actor = node.actor
                choice = node.choice
            actor_choice = [1 if actor == "attention" else 0,
                            1 if actor == "action" else 0]
        else:
            if obs_changed > 0 or self.prev_obs_changed > 0:
                actor_choice = [1, 0]   # attention
            else:
                actor_choice = self.actor_chooser.step([1/2]*2)  # even choice 1/2
            if actor_choice[0] == 1:  # attention
                choice = [1/self.obs_span]*self.obs_span  # even distribution
            else:   # external action
                choice = [1/self.action_dim]*self.action_dim # even distribution
        if actor_choice[0] == 1:  # attention
            action = [-1]*self.action_dim   # no choice
            attention = choice
        else:   # external action
            attention = [-1]*self.obs_span   # no choice
            action = choice
        self.prev_node = node
        self.prev_observation = observation
        self.prev_obs_changed = obs_changed
        self.prev_recog = recog
        self.prev_prev_actor = prev_actor
        self.prev_prev_choice = prev_choice
        self.results['attention'] = attention
        self.results['action'] = action

    def find_node(self, current):
        observations = []
        node = None
        for item in self.episode:
            observations.append(tuple(item[0].tolist()))
        observations.append(tuple(current.tolist()))
        length = len(observations)
        for l in reversed(range(len(observations))):
            index = tuple(observations[length - l -1:])
            if index in self.path_indices:
                # the best node is the beginning node with the max score within the matched subgraphs
                best_node_pair = max(self.path_indices[index], key= lambda x: x[0].success * x[0].success / x[0].total)
                if best_node_pair[0].success / best_node_pair[0].total >= self.min_success_rate:
                    node = best_node_pair[1]    # end node
                    break
        return node

    def recog(self, gated_obs, obs_changed):
        if obs_changed > 0:
            changed = True
        else:
            changed = False
        if max(gated_obs) > 0:
            for i in reversed(self.episode):
                memorized_gated_obs = i[0][:-1]
                if changed and tuple(memorized_gated_obs) == tuple(gated_obs):
                    return 1
        return 0

    def get_executed_actor(self, actions):
        act_key = ""
        actor = None
        for key in actions.keys():
            if max(actions[key]) > 0:
                actor = key
                break
        return actor

    def select_branch(self, branches, observation, actor, choice):
        branch = None
        for br in branches:
            if tuple(br.observation) == tuple(observation) and br.actor == actor \
                    and tuple(br.choice) == tuple(choice):
                branch = br
                break
        return branch

    def update_hypotheses(self, reward):
        branches = self.hypotheses
        self.reward = reward    # for dump
        parent = None
        length = len(self.episode)
        nodes = []
        for i in reversed(range(length)):
            item = self.episode[i]
            observation = item[0]
            actor = item[1]
            choice = item[2]
            if actor is None:
                continue
            branch = self.select_branch(branches, observation, actor, choice)
            if branch is not None:
                branch.total += 1
                branch.success += reward
            else:
                branch = EpisodicMemory.Node(observation, actor, choice, parent)
                branch.total = 1
                branch.success = reward
                branches.append(branch)
            parent = branch
            branches = branch.branches
            nodes.append(branch)
        self.set_path_indices(nodes)

    def set_path_indices(self, nodes):
        nodes.reverse()
        length = len(nodes)
        for begin_pos in range(length):
            for j in range(length - begin_pos):
                end_pos = begin_pos + j
                index = []
                for k in range(end_pos - begin_pos + 1):
                    index.append(tuple(nodes[begin_pos+k].observation))
                index = tuple(index)
                begin_node = nodes[begin_pos]
                end_node = nodes[end_pos]
                if index in self.path_indices:
                    self.path_indices[index].append((begin_node, end_node))
                else:
                    self.path_indices[index] = [(begin_node, end_node)]

    def reset(self):
        self.token = 0
        self.prev_node = None
        self.results['token_out'] = np.array([0])
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.episode = []
        self.on_hypothesis = False
        self.prev_obs_changed = 1
        self.prev_recog = 0
        self.prev_prev_actor = ""
        self.prev_prev_choice = None
        self._task_switch = []  # for debug
        self._feature = []      # for debug
        self._correct = 0       # for debug
        self._recog = False

    def correct_path(self, gated_obs, actor, choice):
        if self.token > 0 and self.token <= self._env["switch_period"] + 1:
            if actor == "attention" and tuple(choice) == (1,0,0,0):
                self._correct = 1
                self._task_switch = gated_obs[:self._env["attention_size"]]
        if self.token > self._env["switch_period"] + 1:
            if self._correct < 1:
                self._correct = 0
            else:
                if self.token <= self._env["switch_period"] + self._env["match_delay"] + self._env["sample_period"] + 1 and \
                        actor == "attention" and tuple(choice[1:-1]) == tuple(self._task_switch):
                    self._correct = 2
                    self._feature = gated_obs
        if self.token > self._env["switch_period"] + self._env["match_delay"] + self._env["sample_period"] + 1:
            if self._correct < 2:
                self._correct = 0
            else:
                if self.token <= self._env["switch_period"] + self._env["match_delay"] *2 + \
                        self._env["sample_period"] + self._env["response_period"] + 1:
                    if actor == "attention" and tuple(choice[1:-1]) == tuple(self._task_switch):
                        self._correct = 3
                        self._recog = (tuple(self._feature) == tuple(gated_obs))
                    if self._correct == 3 and self._recog and actor == "action" and tuple(choice) == (0, 1):
                        self._correct = 4
                    if self._correct == 3 and (not self._recog) and actor == "action" and tuple(choice) == (1, 0):
                        self._correct = 4
        return

    def dump_hypotheses(self, nodes, level, dp):
        for node in nodes:
            for i in range(level):
                dp.write("\t")
            recog = "R" if node.observation[-1] > 0 else ""
            dp.write("{0}{1},{2},{3},{4}/{5}\n".
                       format(node.observation, recog, node.actor, node.choice, int(node.success), node.total))
            self.dump_hypotheses(node.branches, level+1, dp)

    def dump_hypotheses2(self, nodes, line, dp):
        for node in nodes:
            line.append(node)
            if len(node.branches) != 0:   # not terminal
                self.dump_hypotheses2(node.branches, line, dp)
            else:
                if node.success/node.total==1:
                    l = 0
                    for nd in line:
                        for i in range(l):
                            dp.write("\t")
                        recog = "R" if nd.observation[-1] > 0 else ""
                        dp.write("{0}{1},{2},{3},{4}/{5}\n".
                                 format(nd.observation, recog, nd.actor, nd.choice, int(nd.success), nd.total))
                        l += 1
            line = []

    class Node():
        def __init__(self, observation, actor, choice, parent):
            self.parent = parent
            self.branches = []
            self.observation = tuple(observation)
            self.actor = actor
            self.choice = choice
            self.success = 0
            self.total = 0

class Chooser:
    def __init__(self, dim):
        self.dim = dim
        self.rng = np.random.default_rng()   # random generator

    def step(self, distribution):
        distribution = np.array(distribution)
        if sum(distribution) <= 0:
            return np.array([0] * self.dim)
        else:
            distribution = distribution / sum(distribution)
            return np.array(self.rng.multinomial(1, distribution))


class Gate(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.attention_size = config['env']["attention_size"]
        self.atttibute_number = config['env']["attribute_number"]
        self.obs_span = self.atttibute_number + 2   # + task_switch:1 + control:1
        self.obs_dim = self.attention_size * self.obs_span
        self.make_in_port('observation', self.obs_dim)
        self.make_in_port('attention', self.obs_span)
        self.make_in_port('token_in', 1)
        self.make_out_port('gated_obs', self.obs_dim)
        self.make_out_port('attention', self.obs_span)
        self.make_out_port('obs_changed', 1)
        self.make_out_port('token_out', 1)
        self.attention_chooser = Chooser(self.obs_span)
        self.prev_observation = []
        self.prev_gated_obs = []
        self.prev_choice = []

    def fire(self):
        observation = self.get_in_port('observation').buffer
        if len(observation) < self.obs_dim:
            observation = [0] * self.obs_dim # 0s
        given_attention = self.get_in_port('attention').buffer
        if len(given_attention) < self.obs_span:
            given_attention = [1 / self.obs_span] * self.obs_span # even distribution
        given_attention = np.array(given_attention)
        salience = self.salience(observation)
        if sum(salience) == 1:  # only one salient part
            given_attention = salience
        else:
            if max(given_attention) > 0 and max(given_attention) < 1:   # even distribution
                given_attention = salience
            elif max(given_attention) == 1: # hypothesis driven attention
                given_attention = given_attention * salience
            if max(given_attention) == 0: # 0 input
                given_attention = salience
        given_attention = given_attention / sum(given_attention) if sum(given_attention) > 0 else given_attention
        choice = self.attention_chooser.step(given_attention)
        attention = []
        for i in range(self.obs_span):
            for j in range(self.attention_size):
                if choice[i] == 0:
                    attention.append(0)
                else:
                    attention.append(1)
        if tuple(observation) == tuple(self.prev_observation) and \
            max(given_attention) != 1 and len(self.prev_gated_obs)>0:  # not hypothesis driven attention
            gated_obs = self.prev_gated_obs  # use previous output
            choice = self.prev_choice
        else:
            gated_obs = observation * attention
        if tuple(observation) == tuple(self.prev_observation):
            self.results['obs_changed'] = np.array([0])
        else:    # observation changed
            self.results['obs_changed'] = np.array([1])
        self.prev_observation = observation
        self.prev_gated_obs = gated_obs
        self.prev_choice = choice
        self.results['attention'] = choice
        self.results['gated_obs'] = gated_obs

    def salience(self, observation):
        salience = []
        for i in range(self.obs_span):
            salience.append(max(observation[i*self.attention_size:(i+1)*self.attention_size]))
        return np.array(salience)

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.inputs['attention'] = np.array([0] * self.obs_span)
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.prev_observation = []
        self.prev_gated_obs = []
        self.prev_choice = []


class ActionChooser(brica1.brica_gym.Component):
    def __init__(self, config):
        super().__init__()
        self.dim = config['env']['actions']
        self.make_in_port('action', self.dim)
        self.make_in_port('token_in', 1)
        self.make_out_port('action', self.dim)
        self.make_out_port('token_out', 1)
        self.chooser = Chooser(config['env']['actions'])

    def fire(self):
        given_action = self.get_in_port('action').buffer
        if len(given_action) < self.dim:
            given_action = [1/self.dim] * self.dim # even distribution
        action = self.chooser.step(given_action)
        self.results['action'] = np.array(action)

    def reset(self):
        self.token = 0
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_out_port('token_out').buffer = self.results['token_out']


def main():
    parser = argparse.ArgumentParser(description='Minimal Matching to Sample task Agent with Gym in BriCAL')
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                        help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=50, metavar='N',
                        help='Max steps in an episode (default: 50)')
    parser.add_argument('--config', type=str, default='SMA4DM2S.json', metavar='N',
                        help='Model configuration (default: SMA4DM2S.json')
    parser.add_argument('--dump_level', type=int, default=0, help='>=0')
    parser.add_argument('--brical', type=str, default='SMA4DM2S.brical.json', metavar='N',
                        help='a BriCAL json file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    if args.dump is not None:
        try:
            dump = open(args.dump, mode='w')
        except IOError:
            print('Dump path error', file=sys.stderr)
            sys.exit(1)
    else:
        dump = None

    nb = brical.NetworkBuilder()
    f = open(args.brical)
    nb.load_file(f)
    if not nb.check_consistency():
        sys.stderr.write("ERROR: " + args.brical + " is not consistent!\n")
        exit(-1)

    if not nb.check_grounding():
        sys.stderr.write("ERROR: " + args.brical + " is not grounded!\n")
        exit(-1)

    train = {"episode_count": args.episode_count, "max_steps": args.max_steps}
    config['train'] = train

    env = gym.make(config['env']['name'], config=config['env'])

    nb.unit_dic['SMA4DM2S.ActionChooser'].__init__(config)
    nb.unit_dic['SMA4DM2S.Gate'].__init__(config)
    nb.unit_dic['SMA4DM2S.EpisodicMemory'].__init__(config)

    nb.make_ports()

    agent_builder = brical.AgentBuilder()
    model = nb.unit_dic['SMA4DM2S.CognitiveArchitecture']
    agent = agent_builder.create_gym_agent(nb, model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)

    dump_cycle = config["dump_cycle"]
    dump_counter = 0
    reward_sum = 0.0
    # observation: 3 for task switch, 3*2 for patterns; 2 for control
    for i in range(train["episode_count"]):
        last_token = 0
        for j in range(train["max_steps"]):
            scheduler.step()
            current_token = agent.get_out_port('token_out').buffer[0]
            # print("current_token:" + str(current_token) + ", i:" + str(i) + " j:" + str(j))
            if last_token + 1 == current_token:
                last_token = current_token
            if agent.env.done:
                break
        agent.env.flush = True
        if dump is not None and args.dump_level > 2:
            dump.write("On Hypothesis: {0}, reward: {1}\n".
                       format(nb.unit_dic['SMA4DM2S.EpisodicMemory'].on_hypothesis,
                              nb.unit_dic['SMA4DM2S.EpisodicMemory'].reward))
        scheduler.step()
        scheduler.step()
        if dump is not None and args.dump_level > 0:
            reward_sum += agent.get_in_port("reward").buffer[0]
            if dump_counter % dump_cycle == 0 and dump_counter != 0:
                dump.write("{0}: avr. reward: {1:.2f}".
                           format(dump_counter // dump_cycle,
                                  reward_sum / dump_cycle))
                dump.write("\n")
                reward_sum = 0.0
            dump_counter += 1
        nb.unit_dic['SMA4DM2S.Gate'].reset()
        nb.unit_dic['SMA4DM2S.EpisodicMemory'].reset()
        nb.unit_dic['SMA4DM2S.ActionChooser'].reset()
        agent.env.reset()
        # agent.env.out_ports['token_out'] = np.array([0])
        # if agent.env.out_ports['reward'].buffer[0] > 0:
        #     print(str(i) + " reward:" + str(agent.env.out_ports['reward'].buffer[0]))
        agent.env.done = False
    if dump is not None and args.dump_level > 1:
        model.components['SMA4DM2S.EpisodicMemory'].dump_hypotheses(
            model.components['SMA4DM2S.EpisodicMemory'].hypotheses, 0, dump)
    print("Close")
    env.close()


if __name__ == '__main__':
    main()
