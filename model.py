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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class RDQN_multi(nn.Module):
    def __init__(self, args):
        super(RDQN_multi, self).__init__()
        
        self.state_dimension = args.state_dimension
        self.n_hidden = args.n_hidden
        self.n_batches = args.n_batches
        self.dropout = args.drop_out
        
        self.embed_action = nn.Embedding(args.action_embedding_dim, args.n_hidden)
        self.embed_agent = nn.Embedding(args.n_agents, args.n_hidden)

        self.lin_in = nn.Linear(args.state_dimension, args.n_hidden)
        
        self.gru = nn.GRU(args.n_hidden, args.n_hidden, args.rnn_layers, dropout=args.drop_out)

        self.lin_out1 = nn.Linear(args.n_hidden, args.n_hidden)
        self.lin_out2 = nn.Linear(args.n_hidden, args.n_actions)

    def forward(self, h, state, a_id, last_action):

        x = self.lin_in(state)
        x += self.embed_agent(a_id).view(self.n_batches, -1)
        x += self.embed_action(last_action).view(self.n_batches, -1)

        x = x.unsqueeze(0)
        out, ht = self.gru(x, h)
        out = out.view(self.n_batches, -1)
        y = F.relu(self.lin_out1(out))
        Q = self.lin_out2(y)

        return [Q, ht]
