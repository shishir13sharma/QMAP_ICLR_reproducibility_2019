import torch
import torch.nn as nn
import torch.optim as optim

import os
import pdb
import math
import numpy as np

import matplotlib.pyplot as plt
from copy import deepcopy
from gym.utils import seeding
from torch.autograd import Variable

from baselines import logger
from agents.q_map_renderer import Q_Map_Renderer
from qmap.utils.csv_logger import CSVLogger

# While the structure for the DQN and Q-MAP is defined in the models file, the training and choose_action methods for both 
# are defined here in the classes DQN and QMAP. The class Q_Map_DQN_Agent implements the decision tree of the proposed algorithm.
# This class is also responsible for calling the respective optimize methods of DQN and QMAP class and dynamically updates the
# probabilities q_map_goal_proba and random_proba corresponding to choosing Qmap based and random goals respectively.

class DQN():
    def __init__(self, model, n_actions, gamma, lr, replay_buffer, batch_size,
                 optim_iters, grad_norm_clip):
        
        self.model = model
        self.target_model = deepcopy(model)
        self.n_actions = n_actions
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)        
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.optim_iters = optim_iters                                
        self.grad_norm_clip = grad_norm_clip
        
    
    def train(self, obs, acs, rews, obs1, dones, weights, gamma=1.0):
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            obs = obs.to(device)
            acs = acs.to(device)
            rews = rews.to(device).float()
            obs1 = obs1.to(device)
            dones = dones.to(device).float()
            weights = weights.to(device)   
                                
        # Q-values for all actions in the given states        
        qt_values = self.model(obs)                
        
        # Only keeping the Q-values of chosen actions
        qt_selected = qt_values.gather(1, acs.unsqueeze(1)).squeeze()                               
                
        # Q values of all actions from the target model for the next observation 
        qt1_targets = self.target_model(obs1).detach()
        
        # Preserving only the max Q-value for the model target
        _, qt1_selected_acs = qt1_targets.max(-1) 
        qt1_selected_targets = qt1_targets.gather(1, qt1_selected_acs.unsqueeze(1)).squeeze()
        
        qt1_masked_targets = (1.0 - dones) * qt1_selected_targets

        # compute RHS of bellman equation
        qt1_final_target = rews + gamma * qt1_masked_targets
                        
        # Huber loss
        errors = nn.SmoothL1Loss()(qt_selected, qt1_final_target)
        
        # Weights from the priority replay buffer are used
        weighted_errors = torch.mean(weights.float() * errors)        
        
        self.optimizer.zero_grad()
        weighted_errors.backward()
        
        # Gradients are computed using backward method, but are clipped before update
        grad_norms = []
        if self.grad_norm_clip is not None:            
            for params in self.model.parameters():
                grad_norms.append(nn.utils.clip_grad_norm_(params, self.grad_norm_clip))        
            
        self.optimizer.step()
        
        # The TD error is used by the replay buffer to determine priority
        td_errors = qt_selected - qt1_final_target.float()               
        
        return td_errors.detach().cpu().numpy()
        
        
    def act(self, obs):                                               
                
        obs = torch.tensor(obs).permute(0, 3, 1, 2).float()
        batch_size = obs.size()[0]
        if torch.cuda.is_available():
            device = torch.device("cuda")
            obs = obs.to(device)
        q_values = self.model(obs).cpu()
        
        _, deterministic_actions = q_values.max(-1)
                
        return deterministic_actions.detach().cpu().numpy()
        
    def choose_action(self, ob):
        ac = self.act(ob[None])[0]
        return ac

    def optimize(self, t):
        for iteration in range(self.optim_iters):
            samples = self.replay_buffer.sample(self.batch_size, t)            
            obs, acs, rews, obs1, dones, weights = samples
            
            dones = torch.tensor(dones.astype(int))            
            obs = torch.tensor(obs).permute(0, 3, 1, 2).float()
            obs1 = torch.tensor(obs1).permute(0, 3, 1, 2).float()                         
            weights = torch.tensor(weights)
            acs = torch.tensor(acs)
            rews = torch.tensor(rews)
            
            td_errors = self.train(obs, acs, rews, obs1, dones, weights, self.gamma)            
            self.replay_buffer.update_priorities(td_errors)

    def update_target(self):
        # Copies the current DQN's weights to the target DQN as dictated by Double DQN algorithm
        self.target_model.load_state_dict(self.model.state_dict())

class Q_Map():
    
    def __init__(self, model, coords_shape, n_actions, gamma,
                 n_steps, lr, replay_buffer, batch_size, optim_iters, grad_norm_clip
                ):
        
        self.model = model
        self.target_model = deepcopy(model)
        self.coords_shape = coords_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_steps = n_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)              
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size       
        self.optim_iters = optim_iters
        self.grad_norm_clip  = grad_norm_clip
    
    def train(self, obs, acs, target_qt1, weights):
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            obs = obs.to(device)
            acs = acs.to(device)
            target_qt1 = target_qt1.to(device)
            weights = weights.to(device).float()                                   
                
        # Q-values for all actions in the given states        
        qt_values = self.model(obs)        
        
        # Q-values of only chosen actions in the given states are retained        
        qt_selected = torch.gather(qt_values, 1, acs[:,None,None,None].repeat(1, 1, self.coords_shape[0], 
                                                                            self.coords_shape[1])).squeeze()                        
        
        td_errors = qt_selected - target_qt1
        # Huber loss
        losses = nn.SmoothL1Loss()(qt_selected, target_qt1)
        
#         losses = torch.mean(torch.mean(torch.pow(td_errors, 2), 1), 1)                
        
        # Weights from the priority replay buffer are used
        weighted_loss = torch.mean(weights * losses)
        
        self.optimizer.zero_grad()
        
        weighted_loss.backward()                
        
        # Gradients are computed using backward method, but are clipped before update
        grad_norms = []
        if self.grad_norm_clip is not None:            
            for params in self.model.parameters():
                grad_norms.append(nn.utils.clip_grad_norm_(params, self.grad_norm_clip))            
                
        self.optimizer.step()        
        
        # The mean of TD error along the x and y dimensions is used by the replay buffer to determine priority
        errors = torch.mean(torch.mean(torch.abs(td_errors), 1), 1)        
        
        return errors.detach().cpu().numpy()

    def choose_action(self, ob, goal_rc, q_values=None):
                
        if q_values is None:            
            ob = torch.tensor(ob[None]).permute(0, 3, 1, 2).float()
            if torch.cuda.is_available():
                device = torch.device("cuda")
                ob = ob.to(device)
            q_values = self.model(ob)[0].cpu()

        # Greedy action determined by QMAP is chosen
        _, ac = torch.max(q_values[:,goal_rc[0], goal_rc[1]], 0)             

        return ac, q_values

    def _optimize(self, obs, acs, rcw1s, obs1, dones, weights):
                        
        obs = torch.tensor(obs).permute(0, 3, 1, 2).float()
        obs1 = torch.tensor(obs1).permute(0, 3, 1, 2).float()                         
        acs = torch.tensor(acs)
        dones = torch.tensor(dones.astype(int))
        weights = torch.tensor(weights)        
                
        if torch.cuda.is_available():
            device = torch.device("cuda")
            obs = obs.to(device)
            obs1 = obs1.to(device)
            
        # Following code implements the methodology for generating the target for the QMAP using the next observation:
        
        # 1. Forward pass through the network with the next observation
        qt1_values = self.model(obs1)
        target_qt1_values = self.target_model(obs1)        
        
        # 2. Maximize through the depth axis
        _, target_next_acs = torch.max(target_qt1_values, 1)
        mask = np.arange(self.n_actions) == target_next_acs.unsqueeze(3)
        mask = np.swapaxes(np.swapaxes(mask, -1, 1), -1, 2)
        best_qt1_values = qt1_values[mask].reshape(target_next_acs.shape).detach().cpu()
        
        # 3.a) clip the values        
        clipped_best_qt1_values = np.clip(best_qt1_values, 0., 1.) 
        target_qt1_values = clipped_best_qt1_values
                
        window = target_qt1_values.shape[2]
        
        for i in reversed(range(self.n_steps)):
             # 3. b) discount the values
            target_qt1_values *= self.gamma * (1 - dones[:, i, None, None].repeat(1, self.coords_shape[0],
                                                                                  self.coords_shape[1])).float()
            
            # 4. Replace the value at the next coordinate with 1
            rows, cols, delta_ws = rcw1s[:, i, 0], rcw1s[:, i, 1], rcw1s[:, i, 2]
            target_qt1_values[np.arange(self.batch_size), rows, cols] = 1 
            
            # 5. Offset the frame by the required number of pixels for observations with sliding window             
            for j in range(self.batch_size):
                if delta_ws[j] < 0:                   
                    target_qt1_values[j, :, :delta_ws[j]] = target_qt1_values[j, :, -delta_ws[j]:]
                elif delta_ws[j] > 0:                    
                    target_qt1_values[j, :, delta_ws[j]:] = target_qt1_values[j, :, :-delta_ws[j]]
                # target_qs[j, :, :delta_ws[j]] = 0 # WARNING: this is only for forward moving windows like in Mario (can't go back)

        td_errors = self.train(obs, acs, target_qt1_values, weights)        
        return td_errors
    
    def optimize(self, t):
        for iteration in range(self.optim_iters):
            samples = self.replay_buffer.sample_qmap(self.batch_size, t, self.n_steps)
            td_errors = self._optimize(*samples)
            self.replay_buffer.update_priorities_qmap(td_errors)

    def update_target(self):
        # Copies the current QMAP's weights to the target QMAP
        self.target_model.load_state_dict(self.model.state_dict())

        
class Q_Map_DQN_Agent:
    def __init__(self,
        # All                
        n_actions,    
        coords_shape,
        double_replay_buffer,
        task_gamma,
        exploration_schedule,
        seed,
        path,
        learning_starts=1000,
        train_freq=1,
        print_freq=100, 
        renderer_viewer=True,
        # DQN:
        dqn_model=None,
        dqn_lr=5e-4,
        dqn_batch_size=32,
        dqn_optim_iters=1,
        dqn_target_net_update_freq=500,
        dqn_grad_norm_clip=100,
        # QMAP
        q_map_model=None,
        q_map_random_schedule=None,
        q_map_greedy_bias=0.5,
        q_map_timer_bonus=0.5,
        q_map_lr=5e-4,
        q_map_gamma=0.9,
        q_map_n_steps=1,
        q_map_batch_size=32,
        q_map_optim_iters=1,
        q_map_target_net_update_freq=500,
        q_map_min_goal_steps=10,
        q_map_max_goal_steps=20,
        q_map_grad_norm_clip=1000,        
        ):
                
        # All        
        self.n_actions = n_actions      
        self.coords_shape = coords_shape
        self.double_replay_buffer = double_replay_buffer
        self.task_gamma = task_gamma
        self.exploration_schedule = exploration_schedule
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.print_freq = print_freq
        self.use_q_map = True
        self.use_dqn = True
        self.path = path
        self.logger = []    
        exploration_labels = ['steps', 'planned exploration', 'current exploration', 
                              'random actions', 'goal actions', 'greedy actions']
        self.exploration_logger = CSVLogger(exploration_labels, self.path + '/exploration')

        # DQN instantiate
        if dqn_model is not None:            
            self.use_dqn = True
            self.dqn_target_net_update_freq = dqn_target_net_update_freq

            self.dqn = DQN(
                model=dqn_model,                
                n_actions=n_actions,
                gamma=task_gamma,
                lr=dqn_lr,
                replay_buffer=double_replay_buffer,
                batch_size=dqn_batch_size,
                optim_iters=dqn_optim_iters,
                grad_norm_clip=dqn_grad_norm_clip                
            )
        else:
            self.use_dqn = False
            
        # QMAP instantiate
        if q_map_model is not None:            
            self.use_q_map = True
            self.q_map_timer_bonus = q_map_timer_bonus
            self.using_q_map_starts = 2 * self.learning_starts
            self.q_map_random_schedule = q_map_random_schedule
            self.q_map_greedy_bias = q_map_greedy_bias
            self.q_map_goal_proba = 1 # TODO
            self.q_map_gamma = q_map_gamma
            self.q_map_target_net_update_freq = q_map_target_net_update_freq
            self.q_map_min_goal_steps = q_map_min_goal_steps
            self.q_map_max_goal_steps = q_map_max_goal_steps
            self.q_map_min_q_value = q_map_gamma ** (q_map_max_goal_steps-1)
            self.q_map_max_q_value = q_map_gamma ** (q_map_min_goal_steps-1)
            self.q_map_goal = None
            self.q_map_goal_timer = 0
            
            self.q_map = Q_Map(
                model=q_map_model,     
                coords_shape=coords_shape,
                n_actions=n_actions,
                gamma=q_map_gamma,
                n_steps=q_map_n_steps,
                lr=q_map_lr,
                replay_buffer=double_replay_buffer,
                batch_size=q_map_batch_size,
                optim_iters=q_map_optim_iters,
                grad_norm_clip=q_map_grad_norm_clip,                
            )
        else:
            self.use_q_map = False                              
        
        
        self.renderer = Q_Map_Renderer(self.path, viewer=renderer_viewer)
        self.t = 0
        self.episode_rewards = []
        self.random_proba = self.exploration_schedule.value(0)
        self.random_freq = self.exploration_schedule.value(0)
        self.greedy_freq = 1.0 - self.random_freq
        self.goal_freq = 0.0

        if self.use_dqn:
            self.dqn.update_target()

        self.seed(seed)

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Resets the goal and the goal timer
    def reset(self, ob):
        if self.use_q_map:
            self.q_map_goal_timer = 0
            self.q_map_goal = None

        frames = ob[0]
        ac = self.choose_action(ob)        
        
        self.log()
        self.episode_rewards.append(0.0)
        self.prev_ob = ob
        self.prev_ac = ac

        return ac

    def step(self, ob, rew, done):
        prev_frames, (_, _, prev_w), _, _ = self.prev_ob
        frames, (row, col, w), _, _ = ob

        if self.double_replay_buffer is not None:
            self.double_replay_buffer.add(prev_frames, self.prev_ac, rew, (row, col-w, w-prev_w), frames, done)

        self.optimize()

        if not done:
            ac = self.choose_action(ob)            
        else:
            ac = None            
            self.add_to_renderer(ob)
            
        if torch.is_tensor(ac):
            ac = ac.cpu().numpy()

        self.t += 1
        self.episode_rewards[-1] += rew
        self.prev_ob = ob
        self.prev_ac = ac
        
        return ac

    def choose_action(self, ob):
                
        frames, (row, col, w), screen, (full_r, full_c) = ob   
        
        q_map_values = None
        q_map_candidates = []
        q_map_biased_candidates = []            

        if self.use_q_map :
            
            # If the QMAP is used, generate the QMAP values always 
            
            obs = torch.tensor(frames[None]).permute(0, 3, 1, 2).float()        
            if torch.cuda.is_available():
                device = torch.device("cuda")
                obs = obs.to(device)                        
            q_map_values = self.q_map.model(obs)[0].cpu() # (acs, rows, cols)              
                                    
        # Exploration with random action or if Qmap not available
        if self.np_random.rand() < self.random_proba or (not self.use_dqn and self.t <= self.using_q_map_starts):              
            ac = self.np_random.randint(self.n_actions)
            action_type = 'random'

        # Exploration with Qmap
        elif self.use_q_map and self.t > self.using_q_map_starts:                               

            # Case 1: Goal reached                
            if self.q_map_goal_timer > 0 and (row, col) == self.q_map_goal:
                self.q_map_goal_timer = 0
                self.q_map_goal = None

            # Case 2: Goal is unreachable                
            if self.q_map_goal_timer > 0 and self.q_map_goal[1] < w:
                self.q_map_goal_timer = 0 # Set the goal timer to zero and the goal to None
                self.q_map_goal = None

            # Case 3: Trying to reach the goal                
            # Q-Map is queried to take a greedy action in the direction of the goal
            if self.q_map_goal_timer != 0:
                q_map_goal_row, q_map_goal_col = self.q_map_goal
                q_map_goal_col_local = q_map_goal_col - w
                ac, q_map_values = self.q_map.choose_action(frames, (q_map_goal_row, q_map_goal_col_local))
                self.q_map_goal_timer -= 1
                if self.q_map_goal_timer == 0:
                    self.q_map_goal = None
                action_type = 'qmap'

            # Case 4: Goal timer is 0 either due to goal being reached or goal being unreachable
            elif self.q_map_goal_timer == 0:

                # A new goal is chosen with probability εg
                if self.np_random.rand() < self.q_map_goal_proba:
                    
                    # Find a new goal which is not too close and not too far                      
                    q_map_max_values, _ = torch.max(q_map_values, 0) # (rows, cols)
                    q_map_candidates_mask = np.logical_and(self.q_map_min_q_value <= q_map_max_values, self.q_map_max_q_value \
                                                           >= q_map_max_values)
                    q_map_candidates = np.where(q_map_candidates_mask)
                    q_map_candidates = np.dstack(q_map_candidates)[0] # list of (row, col)

                    if len(q_map_candidates) > 0:

                        # Getting biased goals which have the same first step as the greedy action from DQN
                        if self.use_dqn and self.np_random.rand() < self.q_map_greedy_bias:                            
                            greedy_ac = self.dqn.choose_action(frames)
                            q_map_biased_candidates_mask = np.logical_and(
                                q_map_candidates_mask, torch.argmax(q_map_values, 0).detach().cpu().numpy() == greedy_ac)
                            q_map_biased_candidates = np.where(q_map_biased_candidates_mask)
                            q_map_biased_candidates = np.dstack(q_map_biased_candidates)[0] # list of (row, col)

                        # Randomly choosing one of the biased goals
                        if len(q_map_biased_candidates) > 0:
                            goal_idx = self.np_random.randint(len(q_map_biased_candidates))
                            q_map_goal_row, q_map_goal_col_local = q_map_biased_candidates[goal_idx]
                            q_map_expected_steps = math.log(q_map_max_values[q_map_goal_row, q_map_goal_col_local],
                                                            self.q_map_gamma) + 1
                            # 50% bonus to account for interference from random movements 
                            self.q_map_goal_timer = math.ceil(1.5 * q_map_expected_steps) 
                            self.q_map_goal = (q_map_goal_row, q_map_goal_col_local + w)
                            ac = greedy_ac
                            action_type = 'dqn/qmap'

                        # If there are no biased actions to choose from, choose any goal and query Qmap for greedy action
                        else:
                            goal_idx = self.np_random.randint(len(q_map_candidates))
                            q_map_goal_row, q_map_goal_col_local = q_map_candidates[goal_idx]
                            q_map_expected_steps = math.log(q_map_max_values[q_map_goal_row, q_map_goal_col_local],
                                                            self.q_map_gamma) + 1
                            self.q_map_goal_timer = math.ceil((1. + self.q_map_timer_bonus) * q_map_expected_steps)
                            self.q_map_goal = (q_map_goal_row, q_map_goal_col_local + w)
                            ac, q_map_values = self.q_map.choose_action(None, (q_map_goal_row, q_map_goal_col_local),
                                                                        q_map_values) # no need to recompute the Q-Map      
                            action_type = 'qmap'

                        self.q_map_goal_timer -= 1
                        if self.q_map_goal_timer == 0:
                            self.q_map_goal = None

                    # No viable goal candidates could be generated by Qmap. Take a random action
                    else:
                        self.q_map_goal_timer = 0
                        self.q_map_goal = None
                        ac = self.np_random.randint(self.n_actions)
                        action_type = 'random'

                # If neither a random action was chosen(self.random_proba) nor a random goal(self.q_map_goal_proba),
                # then a greedy action is taken using the dqn.
                else:
                    ac = self.dqn.choose_action(frames)
                    action_type = 'dqn'

        else:
            # Exploitation with DQN 
            ac = self.dqn.choose_action(frames)
            action_type = 'dqn'                    
        
        if self.use_q_map:
            q_map_values = q_map_values.detach().cpu().numpy()
            q_map_values = np.moveaxis(q_map_values, 0, -1)                 
        self.add_to_renderer(ob, q_map_values, ac, action_type, q_map_candidates, q_map_biased_candidates)
        
         # Dynamically update exploration probabilities
        if action_type == 'dqn/qmap':
            self.random_freq += 0.01 * (0 - self.random_freq)
            self.greedy_freq += 0.01 * (1 - self.greedy_freq)
            self.goal_freq += 0.01 * (0 - self.goal_freq) 
        elif action_type == 'dqn':
            self.random_freq += 0.01 * (0 - self.random_freq)
            self.greedy_freq += 0.01 * (1 - self.greedy_freq)
            self.goal_freq += 0.01 * (0 - self.goal_freq)
        elif action_type == 'qmap':
            self.random_freq += 0.01 * (0 - self.random_freq)
            self.greedy_freq += 0.01 * (0 - self.greedy_freq)
            self.goal_freq += 0.01 * (1 - self.goal_freq)
        elif action_type == 'random':
            self.random_freq += 0.01 * (1 - self.random_freq)
            self.greedy_freq += 0.01 * (0 - self.greedy_freq)
            self.goal_freq += 0.01 * (0 - self.goal_freq)
        else:
            raise NotImplementedError('unknown action type {}'.format(action_type))

        target_exploration = self.exploration_schedule.value(self.t)
        current_exploration = (1.0 - self.greedy_freq)
        if self.use_q_map and self.t >= self.using_q_map_starts:
            self.random_proba = self.q_map_random_schedule.value(self.t)
            if current_exploration > target_exploration:
                self.q_map_goal_proba -= 0.001
            elif current_exploration < target_exploration:
                self.q_map_goal_proba += 0.001
        else:
            self.random_proba = self.exploration_schedule.value(self.t)
        
        
        if (self.t+1) % 100 == 0:
            self.exploration_logger.log(self.t+1, target_exploration, current_exploration, self.random_freq, self.goal_freq, 
                                        self.greedy_freq)
            
        return ac

    def optimize(self):       
        
        # If learning has started and is to be done for this step, then call the respective optimize functions        
        if (self.use_dqn or self.use_q_map) and self.t >= self.learning_starts and self.t % self.train_freq == 0:
            if self.use_dqn:
                self.dqn.optimize(self.t)

            if self.use_q_map:
                self.q_map.optimize(self.t)

        # For the update freq set for dqn copy the current dqn to the target dqn. (Ref: DDQN)
        if self.use_dqn and self.t >= self.learning_starts and self.t % self.dqn_target_net_update_freq == 0:
            self.dqn.update_target()

        # For the update freq set for qmap model, copy the current qmap model to the target model. (Ref: DDQN)
        if self.use_q_map and self.t >= self.learning_starts and self.t % self.q_map_target_net_update_freq == 0:
            self.q_map.update_target()

        # Save the session        
        dir = self.path + '/Models/'
        if not os.path.isdir(dir):
            os.mkdir(dir)        
        if self.use_dqn  and (self.t+1) % 1e6 == 0:            
            file_name = dir + 'DQN_step_' + str(self.t+1) + '.pt'
            print('saving pytorch DQN', file_name)
            torch.save(self.dqn.model.state_dict(),file_name)

        if self.use_q_map  and (self.t+1) % 1e6 == 0:
            file_name = dir + 'QMAP_step_' + str(self.t+1) + '.pt'
            print('saving pytorch QMAP', file_name)
            torch.save(self.q_map.model.state_dict(),file_name)
    
    def log(self):
        if self.t > 0 and self.print_freq is not None and len(self.episode_rewards) % self.print_freq == 0:
            mean_100ep_reward = np.mean(self.episode_rewards[-100:])
            num_episodes = len(self.episode_rewards)

            logger.record_tabular('steps', self.t)
            logger.record_tabular('episodes', num_episodes)
            logger.record_tabular('mean 100 episode reward', '{:.3f}'.format(mean_100ep_reward))
            logger.record_tabular('exploration (target)', '{:.3f} %'.format(100 * self.exploration_schedule.value(self.t)))
            logger.record_tabular('exploration (current)', '{:.3f} %'.format(100 * (1.0 - self.greedy_freq)))
            logger.dump_tabular()
            
    def load(self, path, num):
               
        if os.path.exists(path + '/Models/DQN_step_' + str(num) + '.pt'):
            self.dqn.model.load_state_dict(torch.load(path + '/Models/DQN_step_' + str(num) + '.pt'))
            self.dqn.model.eval()
        if os.path.exists(path + '/Models/QMAP_step_' + str(num) + '.pt'):
            self.q_map.model.load_state_dict(torch.load(path + '/Models/QMAP_step_' + str(num) + '.pt'))
            self.q_map.model.eval()
        print('model restored :)')
        
    def add_to_renderer(self, ob, q_map_values=None, ac=None, action_type='', q_map_candidates=[], q_map_biased_candidates=[]):
        if self.renderer is not None:
            if self.use_q_map and self.q_map_goal is not None:
                goal = self.q_map_goal
                assert self.q_map_goal_timer > 0
            else:
                goal = None
            self.renderer.add(ob, self.coords_shape, q_map_values, ac, action_type, self.n_actions, q_map_candidates,
                              q_map_biased_candidates, goal)
                  

