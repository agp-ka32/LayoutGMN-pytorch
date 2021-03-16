
import warnings
warnings.simplefilter("ignore")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import get_args
config = get_args()


class MLP(nn.Module):
    def __init__(self, h_sizes):
        '''
        :param h_sizes: a list of hidden layers; the last entry is the size of the output vector
        :return:
        '''
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        assert len(h_sizes) > 1
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))


    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        #output = F.softmax
        out = x
        return out


def build_embeding_layer(vocab_size, dim, drop_prob):
    embed = nn.Sequential(nn.Embedding(vocab_size, dim),
                          nn.ReLU(),
                          nn.Dropout(drop_prob))
    return embed



class GraphEncoder(nn.Module):
    '''Encoder module that projects node and edge features to some embedding'''

    def __init__(self,config, node_hidden_sizes=None,edge_hidden_sizes=None):
        '''

        :param node_hidden_sizes: if provided should be a list of ints, hidden sizes of
        node encoder network, the last element is the size of

        :param edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
        edge encoder network, the last element is the size of the edge outptus

        :param name: name of this module
        '''

        super(GraphEncoder, self).__init__()
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes
        self.drop_prob = 0.5

        num_objs =  12
        self.one_hot_embed = build_embeding_layer(num_objs, config.node_state_dim, self.drop_prob)
        self.MLP_node_geometry = MLP(node_hidden_sizes)
        self.combined_node_feats = MLP([config.node_state_dim*2, config.node_state_dim])
        
        self.MLP_edge = MLP(edge_hidden_sizes)

        if torch.cuda.is_available() and config.cuda:
            self.one_hot_embed = self.one_hot_embed.cuda()
            self.MLP_node_geometry = self.MLP_node_geometry.cuda()
            self.combined_node_feats = self.combined_node_feats.cuda() 

            self.MLP_edge = self.MLP_edge.cuda()




    def forward(self, node_geometry_features, node_room_ids, edge_features=None):
        ''' Encode node and edge features
        :param node_features: [n_nodes, node_feat_dim] float tensor

        :param edge_features: if provided, should be [n_edges, edge_feat_dim] float tensor

        ReturnS:
        node outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings
        edge_outputs: if edge_featres is not None and edge_hidden_sizes is not None, this is
        then a float tensor[n_edges, edge_embedding_dim], edge_embeddings;
        otherwise just the input_edge_features
        '''


        if self._node_hidden_sizes is None: #this is never the case
            node_outputs = node_geometry_features
        else:
            #transposed_node_features = torch.transpose(node_features, 0, 1)
            ############# For label transfer application only ##############
            label_embed = self.one_hot_embed(node_room_ids.long())
            label_embed = label_embed.squeeze(1)
            #label_embed = torch.ones(node_room_ids.shape[0], config.node_state_dim)
            ###############################################################
            geo_embed = self.MLP_node_geometry(node_geometry_features)            
            concat_feat = torch.cat((geo_embed, label_embed), -1)
            
            node_outputs = self.combined_node_feats(concat_feat)


        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            #transposed_edge_features = torch.transpose(edge_features, 0, 1)
            edge_outputs = self.MLP_edge(edge_features)

        return node_outputs, edge_outputs



'''
# example check
num_nodes = 4
node_feature_size = 8
init_node_features = torch.rand(num_nodes, node_feature_size)
node_hidden_sizes = [8,16,32]
edge_hidden_sizes = [8,16,32]


mlp = MLP([4, 16])
print(list(mlp.parameters()))


my_obj = GraphEncoder(node_hidden_sizes, edge_hidden_sizes)
print(list(my_obj.parameters()))
'''
