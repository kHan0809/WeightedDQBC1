import copy

import gym
import torch

from Model.class_model import  BC_agent
from Utils.arguments import get_args
import d4rl
from Utils.utils import d4rl_dataset, Eval, logger
import os
import numpy as np

log_dir = os.getcwd()+'/log'
task_list = ["halfcheetah", "walker2d", "hopper"]
data_list = ["-medium-v2", "-medium-expert-v2","-medium-replay-v2","-random-v2"]


file_names = os.listdir(log_dir)
total_name,target_epoch_list = [], []
for data in data_list:
  for task in task_list:
    for file_name in file_names:
      max_score, max_epoch, count = -10, 0, 0
      if task+data in file_name:
        file = open(log_dir+'/'+file_name)
        while True:
          line = file.readline()
          line_list = line.split()
          count += 1
          if len(line_list) > 2 and count > 6:
            if float(line_list[2]) > max_score:
              max_score, max_epoch = float(line_list[2]), int(line_list[1])
          if not line:
            break
        total_name.append(file_name)
        target_epoch_list.append(max_epoch)
print(total_name)
print(target_epoch_list)

args = get_args()

for idx,task in enumerate(total_name):
  print(task)
  log = logger(task[:-6], './log_qbc', iter=idx) #target_epoch_list[idx]+
  env = gym.make(task[:-11])

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  action_max = env.action_space.high[0]
  epi_length = env.spec.max_episode_steps

  agent = BC_agent(state_dim, action_dim, args)
  agent.init_pi(task[:-6], int(task[-5]), target_epoch_list[idx], "./model_save/qbc")
  mean_, min_, max_ = Eval(env, agent, target_epoch_list[idx], args)
  log.write_eval(target_epoch_list[idx], mean_, min_, max_)
