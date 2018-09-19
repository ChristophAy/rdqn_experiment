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
import numpy as np
import pdb

class env_multi():
    
    def __init__(self, args):
        
        self.T = args.T
        self.var = args.var
        self.n_states = args.n_states
        self.n_batches = args.n_batches
        self.n_agents = args.n_agents
        network = self.load_network(args.network_path + args.network_file)
        self.neighborhoods = []
        for j in range(self.n_agents):
            neighborhood = network[j]
            self.neighborhoods.append([neighborhood[i] for i in range(len(neighborhood))])
        
        self.reset()
        
    def reset(self):
        
        self.time_step = 0
        self.ws = np.random.randint(0, self.n_states, size=self.n_batches)
        errs = torch.randn(self.n_batches, self.n_agents) * np.sqrt(self.var)
        ws_list = [self.ws for j in range(self.n_agents)]
        self.world = torch.LongTensor(ws_list).t()
        self.signals = errs + self.world.float()
        
        tmp = -1 * np.ones((self.n_batches, 1), dtype=int)
        self.last_actions = [torch.LongTensor(tmp) for i in range(self.n_agents)]
                
    def step(self, actions):
        
        self.last_actions = actions

        rewards = []
        
        for i, action in enumerate(actions):
        
            reward = torch.zeros(self.n_batches, 0)
            reward[action[:, 0] == self.world[:, 0]] = 1
            rewards.append(reward)
            
        self.time_step += 1
        
        return rewards
    
    def get_state(self, agent):
        time = self.time_step * torch.FloatTensor(np.ones((self.n_batches)))
        a_id = agent.agent_id * torch.LongTensor(np.ones((self.n_batches), dtype=int))
        state = torch.FloatTensor(np.zeros((self.n_batches, self.n_agents + 2)))

        for i, la in enumerate(self.last_actions):
            if i in self.neighborhoods[agent.agent_id]:
                state[:, i] = la + 1

        state[:, self.n_agents] = self.signals[:, agent.agent_id]
        state[:, self.n_agents + 1] = time

        return [state, a_id, self.last_actions[agent.agent_id] + 1]

    def load_network(self, filename):

        f = open(filename, 'rb')
        neighborhoods = dict() 
        cnt = 0
        for l in f:
            tmp = [int(i) for i in l.split(',')]
            neighborhoods[cnt] = tmp[1:]
            cnt += 1

        return neighborhoods

