import copy
import collections
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
sys.path.append('..')
from utils.create_agent import createAgent
from utils.parameters import *
from utils.env_wrapper import EnvWrapper
from networks import point_net
from agents.dqn_agent_com import DQNAgentCom

env_config = {'render': False, 'obs_type': 'point_cloud'}
env = EnvWrapper(0, 'pybullet', 'block_picking', env_config, planner_config)

def test():
    env.reset()
    action = env.getNextAction()
    dim_action = 54
    states, obs, rewards, dones = env.step(action)
    print(states.shape)

    model = point_net.PointQNet(dim_action)
    model.eval()
    next_action, tmat3, tmat64 = model(obs.unsqueeze(0))
    n_p = 2
    n_theta = 1
    agent = DQNAgentCom(lr=lr, gamma=gamma, device='cuda', dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta)
    agent.initNetwork(model)

def train():
    env.reset()


if __name__ == "__main__":
    test()
