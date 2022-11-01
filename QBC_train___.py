import copy

import gym
import torch

from Model.class_model import  BC_agent
from Utils.arguments import get_args
import d4rl
from Utils.utils import d4rl_dataset, Eval, logger

data_list = ["-medium-v2", "-medium-expert-v2","-medium-replay-v2","-random-v2"]
args = get_args()

for data_ in data_list:
  task=args.task_name + data_
  print("[TASK] : " + task)
  env = gym.make(task)
  dataset = d4rl_dataset(env.unwrapped)
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  action_max = env.action_space.high[0]
  epi_length = env.spec.max_episode_steps

  epoch_count, max_return, max_idx = 0, 0, 0
  agent = BC_agent(state_dim, action_dim, args)

  for num in range(args.qbc_train_epoch):
    for step in range(args.train_num_per_epoch):
      batch = dataset.get_data()
      agent.train_QBC(batch)
    epoch_count += 1

    # ====Eval====
    if epoch_count % args.eval_period == 0:
      mean_, min_, max_ = Eval(env, agent, epoch_count, args)




      # if mean_ > max_return:
      #   max_return, max_idx = copy.deepcopy(mean_), copy.deepcopy(epoch_count)
      #   agent.save_checkpoint(args.task_name+str(args.cql), epoch_count, "./model_save/qbc")