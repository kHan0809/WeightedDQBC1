import copy

import gym
import torch

from Model.class_model import  BC_agent
from Utils.arguments import get_args
import d4rl
from Utils.utils import d4rl_dataset, Eval, logger

task_list = ["hopper"]#,"walker2d","halfcheetah"]#,
data_list = ["-medium-expert-v2"]#,"-medium-replay-v2","-medium-v2","-random-v2"]#
args = get_args()

for task_ in task_list:
  for data_ in data_list:
    task=task_ + data_
    print("[TASK] : " + task)
    env = gym.make(task)
    dataset = d4rl_dataset(env.unwrapped)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0]
    epi_length = env.spec.max_episode_steps
    for iteration in range(1):
      log = logger(task+str(args.cql),'./log',iter=iteration)

      agent = BC_agent(state_dim,action_dim,args)
      agent.__init__(state_dim,action_dim,args) # since

      epoch_count = 0
      #==========BC train====================
      log.write_sep("BC")
      for num in range(args.bc_train_epoch):
        for step in range(args.train_num_per_epoch):
          batch = dataset.get_data()
          agent.train_bc(batch)
        epoch_count += 1

        #====Eval====
        if epoch_count % args.eval_period == 0:
          mean_, min_, max_ = Eval(env,agent,epoch_count,args)
          log.write_eval(epoch_count,mean_,min_,max_)
          agent.save_checkpoint(task+str(args.cql), iteration, epoch_count, "./model_save/bc")

      agent = BC_agent(state_dim, action_dim, args)
      #=============Q train=====================
      log.write_sep("Q")
      agent.init_pi(task+str(args.cql),iteration, 100, "./model_save/bc")
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
          agent.save_checkpoint(task+str(args.cql), iteration, 0, "./model_save/bc_q")

      #==========QBC train====================
      log.write_sep("QBC")
      epoch_count, max_return, max_idx = 0, 0, 0
      agent = BC_agent(state_dim,action_dim,args)
      agent.__init__(state_dim, action_dim, args)
      agent.init_q(task+str(args.cql),  iteration, 0, "./model_save/bc_q")

      for num in range(args.qbc_train_epoch):
        for step in range(args.train_num_per_epoch):
          batch = dataset.get_data()
          agent.train_QBC(batch)
        epoch_count += 1

        #====Eval====
        if epoch_count % args.eval_period == 0:
          mean_, min_, max_  = Eval(env,agent,epoch_count,args)
          log.write_eval(epoch_count, mean_, min_, max_)
          agent.save_checkpoint(task + str(args.cql), iteration, epoch_count, "./model_save/qbc")


      # if mean_ > max_return:
      #   max_return, max_idx = copy.deepcopy(mean_), copy.deepcopy(epoch_count)
      #   agent.save_checkpoint(args.task_name+str(args.cql), epoch_count, "./model_save/qbc")