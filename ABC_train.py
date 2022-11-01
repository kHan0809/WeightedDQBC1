import copy

import gym
import torch

from Model.class_model import  ABC_agent
from Utils.arguments import get_args
import d4rl
from Utils.utils import d4rl_dataset, Eval, logger

args = get_args()

env = gym.make(args.task_name)
dataset = d4rl_dataset(env.unwrapped)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
epi_length = env.spec.max_episode_steps
log = logger("ABC_"+args.task_name+str(args.cql),'./log',iter=0)

agent = ABC_agent(state_dim,action_dim,args)
# epoch_count, max_return, max_idx = 0, -100, 0
# #==========BC train====================
# log.write_sep("BC")
# for num in range(args.bc_train_epoch):
#   for step in range(args.train_num_per_epoch):
#     batch = dataset.get_data()
#     agent.train_bc(batch)
#   epoch_count += 1
#
#   #====Eval====
#   if epoch_count % args.eval_period == 0:
#     mean_, min_, max_ = Eval(env,agent,epoch_count,args)
#     log.write_eval(epoch_count,mean_,min_,max_)
#     agent.save_checkpoint("ABC_"+args.task_name+str(args.cql), epoch_count, "./model_save/bc")

#=============Q train=====================
log.write_sep("Q")
agent.init_pi(args.task_name+str(args.cql), 80, "./model_save/bc")
# agent.init_pi("ABC_"+args.task_name+str(args.cql), 80, "./model_save/bc")
epoch_count = 0
for num in range(args.q_train_epoch):
  for step in range(args.train_num_per_epoch):
    batch = dataset.get_data()
    agent.train_Q(batch,args.cql)

  epoch_count += 1
  if epoch_count % args.eval_period == 0:
    q1, q2 = agent.test_q(batch)
    log.write_q(epoch_count,torch.mean(q1).cpu().detach().numpy(),torch.mean(q2).cpu().detach().numpy())
    print("[epoch] :", epoch_count, "Q1 : ", sum(q1) / batch[0].shape[0], "Q2 : ", sum(q2) / batch[0].shape[0])
    agent.save_checkpoint("ABC_"+args.task_name+str(args.cql), args.q_idx, "./model_save/bc_q")

#==========QBC train====================
log.write_sep("ABC")
epoch_count, max_return, max_idx = 0, 0, 0
agent = ABC_agent(state_dim,action_dim,args)
agent.init_q("ABC_"+args.task_name+str(args.cql),  args.q_idx, "./model_save/bc_q")

for num in range(args.qbc_train_epoch):
  for step in range(args.train_num_per_epoch):
    batch = dataset.get_data()
    agent.train_ABC(batch)
  epoch_count += 1

  #====Eval====
  if epoch_count % args.eval_period == 0:
    mean_, min_, max_  = Eval(env,agent,epoch_count,args)
    log.write_eval(epoch_count, mean_, min_, max_)
    if mean_ > max_return:
      max_return, max_idx = copy.deepcopy(mean_), copy.deepcopy(epoch_count)
      agent.save_checkpoint("ABC_"+args.task_name+str(args.cql), epoch_count, "./model_save/qbc")
print("[QBC MAX_IDX] : ", max_idx)

#random
# mean : 500

#medium-expert
# ==================[Eval]====================
# Epoch        :  995
# Mean return  :  11050.88704032498 Min return 10798.023899182852 Max return 11368.050196417964
# Saving model to ./model_save/qbc/halfcheetah-medium-expert-v2True-995.pt