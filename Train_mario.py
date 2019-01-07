from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines.common.schedules import LinearSchedule

import argparse
import scipy.misc
import os, datetime, time, re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import gym
import matplotlib.pyplot as plt
from IPython import display
from copy import deepcopy
from agents.models_pytorch import dqn_model, qmap_model
from agents.q_map_dqn_agent_pytorch import Q_Map_DQN_Agent

from agents.replay_buffers import DoublePrioritizedReplayBuffer
from envs.custom_mario import CustomSuperMarioAllStarsEnv
from envs.wrappers import PerfLogger
from time import gmtime, strftime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='random number generator seed', type=int, default=0)
parser.add_argument('--path', default='pytorch_results/' + strftime("%d_%b_%Y_%H_%M_%S", gmtime()))
parser.add_argument('--level', help='game level', default='1.1')
parser.add_argument('--load', help='steps of the models saved in Models folder', default=None)
boolean_flag(parser, 'dqn', default=True)
boolean_flag(parser, 'qmap', default=True)
boolean_flag(parser, 'render', help='play the videos', default=False)
args = parser.parse_args()

env = CustomSuperMarioAllStarsEnv(screen_ratio=4, coords_ratio=8, use_color=False, use_rc_frame=False,
                                  stack=3, frame_skip=2, action_repeat=4, level=args.level)

coords_shape = env.coords_shape
set_global_seeds(args.seed)
env.seed(args.seed)

mario_dqn = dqn_model(
        observation_space=env.observation_space.shape,
        conv_params=np.array([(32, 8, 2, 3), (32, 6, 2, 2), (64, 4, 2, 1)]),
        hidden_params=np.array([1024]),        
        layer_norm=True,
        activation_fn = F.relu,
        n_actions=env.action_space.n
    )

print(mario_dqn)

mario_qmap = qmap_model(
        observation_space=env.observation_space.shape,
        conv_params=np.array([(32, 8, 2, 3), (32, 6, 2, 2), (64, 4, 2, 1)]),
        hidden_params=np.array([1024]),
        deconv_params = np.array([(64, 4, 2, 1), (32, 6, 2, 2), (env.action_space.n, 4, 1, 2)]),
        layer_norm=True,
        activation_fn = F.elu,        
    )

print(mario_qmap)

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
mario_dqn = mario_dqn.to(device)
mario_qmap = mario_qmap.to(device)

n_steps = int(5e6)
exploration_schedule = LinearSchedule(schedule_timesteps=n_steps, initial_p=1.0, final_p=0.05)
q_map_random_schedule = LinearSchedule(schedule_timesteps=n_steps, initial_p=0.1, final_p=0.05)
double_replay_buffer = DoublePrioritizedReplayBuffer(int(5e5), alpha=0.6, epsilon=1e-6, 
                                                     timesteps=n_steps, initial_p=0.4, final_p=1.0)
task_gamma = 0.99


agent = Q_Map_DQN_Agent(    
    n_actions=env.action_space.n, 
    coords_shape=env.unwrapped.coords_shape,
    double_replay_buffer=double_replay_buffer,
    task_gamma=task_gamma,
    exploration_schedule=exploration_schedule,
    seed=args.seed,
    path=args.path,
    learning_starts=1000,
    train_freq=4,
    print_freq=1,    
    renderer_viewer=True,
    # DQN
    dqn_model= mario_dqn,
    dqn_lr=1e-4,
    dqn_optim_iters=1,
    dqn_batch_size=32,
    dqn_target_net_update_freq=1000,
    dqn_grad_norm_clip=1000,
    #QMAP
    q_map_model=mario_qmap,
    q_map_random_schedule=q_map_random_schedule,
    q_map_greedy_bias=0.5,
    q_map_timer_bonus=0.5, # 50% more time than predicted
    q_map_lr=3e-4,
    q_map_gamma=0.9,
    q_map_n_steps=1,
    q_map_batch_size=32,
    q_map_optim_iters=1,
    q_map_target_net_update_freq=1000,
    q_map_min_goal_steps=15,
    q_map_max_goal_steps=30,
    q_map_grad_norm_clip=1000
)
if args.load is not None:
    agent.load(args.path, args.load)

env = PerfLogger(env, agent.task_gamma, agent.path)
done = True
episode = 0
score = None
best_score = -1e6
best_distance = -1e6
previous_time = time.time()
last_ips_t = 0

for t in range(n_steps+1):        
    
    if done:
        new_best = False
        if episode > 0:
            if score >= best_score:
                best_score = score
                new_best = True
            distance = env.unwrapped.full_c
            if distance >= best_distance:
                best_distance = distance
                new_best = True

        if episode > 0 and (episode < 50 or episode % 10 == 0 or new_best):
            current_time = time.time()
            ips = (t - last_ips_t) / (current_time - previous_time)
            print('step: {} IPS: {:.2f}'.format(t+1, ips))
            name = 'score_%08.3f'%score + '_distance_' + str(distance) + '_steps_' + str(t+1) + '_episode_' + str(episode)                        
            agent.renderer.render(name)
            previous_time = current_time
            last_ips_t = t
        else:            
            agent.renderer.reset()
            
        episode += 1
        score = 0

        ob = env.reset()
        ac = agent.reset(ob)
        
    ob, rew, done, _ = env.step(ac)        
    score += rew                            
    
    ac = agent.step(ob, rew, done)    

env.close()

