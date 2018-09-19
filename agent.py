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

import torch
from torch.autograd import Variable
from copy import deepcopy
import numpy as np
import pdb

class agent():
    
    def __init__(self, args, agent_id=0):

        self.epsilon = args.epsilon
        self.n_actions = args.n_actions
        self.n_batches = args.n_batches
        self.gamma = args.gamma
        self.last_action = torch.LongTensor(np.ones((self.n_batches, 1), dtype=int))
        self.agent_id = agent_id
        self.rnn_layers = args.rnn_layers
        
    def setup_models(self, model, target):
    
        self.model = model
        self.target = target
        self.initHidden()

    def initHidden(self):

        self.Q_s = []
        self.Qt_s = []
        self.actions = []
        self.rewards = []

        self.h = Variable(torch.zeros(self.rnn_layers, self.n_batches, self.model.n_hidden))
        self.h_target = Variable(torch.zeros(self.rnn_layers, self.n_batches, self.model.n_hidden), volatile=True)

    def set_optimizer(self, optimizer):

        self.optimizer = optimizer
        
    def take_action(self, state, a_id, last_action, test=False):

        [Q, self.h] = self.model(self.h, Variable(state), Variable(a_id), Variable(last_action))
        self.Q_s.append(Q)
        [Qt, self.h_target] = self.target(self.h_target, Variable(state), Variable(a_id), Variable(last_action))
        self.Qt_s.append(Qt)
        
        if not test:
            action = torch.LongTensor(np.random.randint(0,self.n_actions, size=(self.n_batches, 1)))
            r_a = torch.FloatTensor(np.random.uniform(0, 1, size=(self.n_batches, 1)))
            action[r_a >= self.epsilon] = Q.data.max(1)[1][r_a >= self.epsilon]
        else:
            action = torch.LongTensor(np.random.randint(0,self.n_actions, size=(self.n_batches, 1)))
            action[:, 0] = Q.data.max(1)[1]

        self.last_action = action
        self.actions.append(action)
        
        return [Q, action]
    
    def reset_target(self):
        
        self.target = deepcopy(self.model)