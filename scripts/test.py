import copy
import collections
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('../helping_hands_rl_envs')
from utils.create_agent import createAgent
from utils.parameters import *
from utils.env_wrapper import EnvWrapper

def test():
    plt.style.use('default')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    agent = createAgent(test=True)
    agent.train()
    # agent.loadModel(load_model_pre)
    states, obs = envs.reset()
    test_episode = 1000
    total = 0
    s = 0
    step_times = []
    pbar = tqdm(total=test_episode)
    while total < 1000:
        actions_star_idx, actions_star = agent.getGreedyActions(states, obs)
        states_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)

        states = copy.copy(states_)
        obs = copy.copy(obs_)

        s += rewards.sum().int().item()

        if dones.sum():
            total += dones.sum().int().item()

        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total, float(s) / total if total != 0 else 0)
        )
        pbar.update(dones.sum().int().item())

if __name__ == '__main__':
    test()
