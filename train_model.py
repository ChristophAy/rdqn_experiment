# author: christoph aymanns

# Python implementation of:

#     Fake News in Social Networks

#     @article{aymanns2017fake,
#       title={Fake News in Social Networks},
#       author={Aymanns, Christoph and Foerster, Jakob and Georg, Co-Pierre},
#       journal={arXiv preprint arXiv:1708.06233},
#       year={2017}
#     }

# Based on:

#     Learning to Communicate with Deep Multi-Agent Reinforcement Learning

#     @article{foerster2016learning,
#         title={Learning to Communicate with Deep Multi-Agent Reinforcement Learning},
#         author={Foerster, Jakob N and Assael, Yannis M and de Freitas, Nando and Whiteson, Shimon},
#         journal={arXiv preprint arXiv:1605.06676},
#         year={2016}
#     }


from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch.optim as optim
import pdb
from copy import deepcopy
import torch.nn.functional as F
import pickle
import random
import string
import argparse

import agent
import env
import model as rnn

reload(agent)
reload(env)
reload(rnn)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def test(players, game, args, n_test=32):

    average_reward = 0
    reward_ts = np.zeros((args.T, n_test))

    for i in range(n_test):
        average_loss, average_reward, reward_ts[:, i] = run_episode(players, game, args, 0, average_reward, test=True)

    average_reward =  average_reward / (args.n_agents * n_test * args.T)

    return average_reward, reward_ts.mean(axis=1)

def run_episode(players, game, args, average_loss, average_reward, test=False, optimizer=None, criterion=None):

    for p in players:
        p.initHidden()

    game.reset()
    reward_ts = np.zeros(args.T)

    for t in range(args.T):
            
        # All agents act
        actions = []
        for j in range(args.n_agents):
            [state, a_id, la] = game.get_state(players[j])
            [Q_model, action] = players[j].take_action(state, a_id, la, test=test)
            actions.append(action)
            
        # Compute rewards given agents actions
        rewards = game.step(actions)

        for j in range(args.n_agents):
            players[j].rewards.append(rewards[j])
            average_reward += rewards[j].numpy().mean()
            reward_ts[t] += rewards[j].numpy().mean() / args.n_agents

    if not test:
        if not args.weight_sharing:
            for j in range(args.n_agents):
                players[j].optimizer.zero_grad()
                loss = 0
                # compute TD errors
                for t in range(args.T):
                    Q = players[j].Q_s[t]
                    a = players[j].actions[t]
                    r = players[j].rewards[t]
                    Q_sa = Q.gather(1, Variable(a))[:, 0]
                    if t < args.T - 1:
                        Qt = players[j].Qt_s[t + 1]
                        td = r + players[j].gamma * Qt.data.max(1)[0]
                    else:
                        td = r

                    loss_out = criterion(Q_sa, Variable(td, requires_grad=False))
                    average_loss += loss_out.data.numpy()[0]
                    loss += loss_out

                loss.backward()

                players[j].optimizer.step()
        else:
            optimizer.zero_grad()
            loss = 0
            # compute TD errors
            for t in range(args.T):
                for j in range(args.n_agents):
                    Q = players[j].Q_s[t]
                    a = players[j].actions[t]
                    r = players[j].rewards[t]
                    Q_sa = Q.gather(1, Variable(a))[:, 0]
                    if t < args.T - 1:
                        Qt = players[j].Qt_s[t + 1]
                        td = r + players[j].gamma * Qt.data.max(1)[0]
                    else:
                        td = r

                    loss_out = criterion(Q_sa, Variable(td, requires_grad=False))
                    average_loss += loss_out.data.numpy()[0]
                    loss += loss_out

            loss.backward()

            optimizer.step()


    return average_loss, average_reward, reward_ts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n_hidden', help='GRU hidden units', type=int, default=12)
    parser.add_argument('-rnn_layers', help='GRU layers', type=int, default=2)
    parser.add_argument('-drop_out', help='GRU dropout', type=float, default=0)
    parser.add_argument('-learning_rate', help='learning rate', type=float, default=5e-4)
    parser.add_argument('-learning_momentum', help='RMS prop momentum', type=float, default=0.05)

    parser.add_argument('-n_actions', help='number of actions', type=int, default=2)
    parser.add_argument('-n_states', help='number of states', type=int, default=2)
    parser.add_argument('-n_batches', help='number of batches', type=int, default=12)
    parser.add_argument('-n_agents', help='number of agents', type=int, default=10)
    parser.add_argument('-var', help='variance of signal', type=float, default=1)
    parser.add_argument('-T', help='number of time steps', type=int, default=10)
    parser.add_argument('-gamma', help='discount factor', type=float, default=0.99)

    parser.add_argument('-epsilon', help='exploration rate', type=float, default=0.05)
    parser.add_argument('-epsilon_min', help='min exploration rate', type=float, default=0.0001)
    parser.add_argument('-epsilon_decay_rate', help='exploration rate decay rate', type=float, default=0.999)
    parser.add_argument('-record_every', help='reward recording interval', type=int, default=100)
    parser.add_argument('-target_reset', help='dqn reset frequency', type=int, default=100)
    parser.add_argument('-smooth', help='moving average parameter', type=float, default=0.5)
    parser.add_argument('-n_iterations', help='number of training episodes', type=int, default=10000)
    parser.add_argument('-n_eval', help='number of test episodes during training', type=int, default=10)
    parser.add_argument('-n_eval_final', help='number of test episodes after training', type=int, default=1000)

    parser.add_argument('-network_path', help='location of network file', default='social_network_files/')
    parser.add_argument('-network_file', help='network file', default='social_network_complete.txt')
    parser.add_argument('-fname', help='file to load', default='')

    parser.add_argument('-eps_decay', help='decay of exploration rate', action="store_true")
    parser.add_argument('-load_model', help='load model from file', action="store_true")
    parser.add_argument('-weight_sharing', help='use weight sharing', action="store_true")

    args = parser.parse_args()

    args.state_dimension = args.n_agents + 2
    args.action_embedding_dim = args.n_actions + 1

    if args.load_model:

        fname = 'results/' + args.fname

        [players, game, args] = pickle.load(open(fname, 'rb'))

        test_reward, reward_ts = test(players, game, T, n_agents, n_test=args.n_eval_final)

        print(reward_ts)

        f, ax = plt.subplots(1, 1)
        ax.plot(reward_ts)

        plt.show()

    else:

        if args.weight_sharing:
            model = rnn.RDQN_multi(args)
            target = rnn.RDQN_multi(args)
            optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)

        game = env.env_multi(args)
        players = []
        for i in range(args.n_agents):
            player = agent.agent(args, agent_id=i)
            if args.weight_sharing:
                player.setup_models(model, target)
            else:
                model = rnn.RDQN_multi(args)
                target = rnn.RDQN_multi(args)
                optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.learning_momentum)
                player.setup_models(model, target)
                player.set_optimizer(optimizer)

            players.append(player)

        criterion = nn.MSELoss()
        average_loss = 0
        average_reward = 0
        loss_record = []
        reward_record = []
        loss_ma = 0
        tr_reward_ma = 0
        te_reward_ma = 0

        fname = 'results/post_training_state_' + id_generator()


        for i in range(args.n_iterations):

            if args.eps_decay:
                for p in players:
                    p.epsilon = np.max([args.epsilon_min, p.epsilon * args.epsilon_decay_rate])

            if args.weight_sharing:
                average_loss, average_reward, reward_ts = run_episode(players, game, args, average_loss,
                                                                      average_reward, optimizer=optimizer, criterion=criterion)
            else:
                average_loss, average_reward, reward_ts = run_episode(players, game, args, average_loss,
                                                                      average_reward, criterion=criterion)
                
            if i % args.target_reset == 0:
                for j in range(args.n_agents):
                    players[j].reset_target()
                
            if i % args.record_every == 0:
                test_reward, reward_ts = test(players, game, args, n_test=args.n_eval)
                loss_record.append(average_loss / (args.record_every * args.n_agents * args.T))
                reward_record.append(average_reward / (args.record_every * args.n_agents * args.T))
                if i == 0:
                    tr_reward_ma =  reward_record[-1]
                    te_reward_ma = test_reward
                    loss_ma = loss_record[-1]
                else:
                    tr_reward_ma = args.smooth * tr_reward_ma + (1 - args.smooth) * reward_record[-1]
                    te_reward_ma = args.smooth * te_reward_ma + (1 - args.smooth) * test_reward
                    loss_ma = args.smooth * loss_ma + (1 - args.smooth) * loss_record[-1]
                print('step %d, eps %f, average loss = %f, average training reward = %f, test reward = %f' % (i, players[0].epsilon, loss_ma, tr_reward_ma, te_reward_ma))
                average_loss = 0
                average_reward = 0

                pickle.dump([players, game, args], open(fname, 'wb'))

        pickle.dump([players, game, args], open(fname, 'wb'))

        test_reward, reward_ts = test(players, game, args, n_test=args.n_eval_final)

        f, ax = plt.subplots(1, 1)
        ax.plot(reward_ts)

        plt.show()