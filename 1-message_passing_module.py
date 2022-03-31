
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from graph_encoder_0 import *

class the_aggregation_module(object):
    '''
    pytorch equivalent of tf.unsorted_segment_sum
    '''
    def __init__(self, messages, to_idx, n_nodes):
        '''

        :param messgaes: [n_edges, edge_message_dim] tensor

        :param to_idx: [n_edges] tensor, the index
        of the to nodes, i.e. where each message should go to

        :param n_nodes: int, which is the number
        of nodes to aggregate into

        return:
            tensor [n_nodes, edge_emb_dim]
        '''


        super(the_aggregation_module, self).__init__()
        self.messages = messages
        self.to_idx = to_idx
        self.n_nodes = n_nodes

        #self._main()

    def _main(self):
        #dim = len(self.to_idx.shape)
        aggregated_sum = scatter_sum(self.messages, self.to_idx, dim=0)

        assert aggregated_sum.shape[0] == self.n_nodes

        return aggregated_sum



def graph_prop_once(node_states, from_idx, to_idx, message_net,
                    aggregation_module, edge_features=None):
    ''' One round of message passing in a graph

    :param node_states: [n_nodes, node_state_dim] float tensor,
    node state vectors, one row for each node; [n_nodes, 32]

    :param from_idx: [n_edges] int tensor, index of the "from nodes"

    :param to_idx: [n_edges] int tensor, index of the "to nodes"

    :param message_net: a network for the edges; an MLP in our case

    :param aggregation_module: a module that aggregates messages on edges
    to aggregated messages for each node.  Should be callable and can be
    called like the following,
        `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
      where messages is [n_edges, edge_message_dim] tensor, to_idx is a [n_edges]
      tensor the index of the to nodes, i.e. where each message should go to,
      and n_nodes is an int which is the number of nodes to aggregate into.

    :param edge_features: if provided, should be a [n_edges, edge_emb_dim] float
      tensor, extra features for each edge


    Returns:
    aggregated_messages: an [n_nodes, edge_emb_dim] float tensor, the
      aggregated messages, one row for each node
    '''

    from_states = node_states[from_idx]
    to_states = node_states[to_idx]

    edge_inputs = [from_states, to_states]
    if edge_features is not None:
        edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)

    messages = message_net(edge_inputs) # output = [n_edges, 64]

    return aggregation_module(messages, to_idx, node_states.shape[0])._main()


class GraphPropLayer(nn.Module):
    '''Implementation of a Graph propagation layer'''

    def __init__(self,
                message_net,
                reverse_message_net,
                node_update_MLP,
                node_state_dim,
                edge_hidden_sizes,
                node_hidden_sizes,
                #edge_net_init_scale=0.1,
                node_update_type='residual',
                use_reverse_direction=True,
                reverse_dir_param_different=True,
                layer_norm=False):
        '''
        :param message_net: a network for the edges; an MLP in our case
        :param node_state_dim: int, dimensionality of node states = 32

        :param edge_hidden_sizes: list of ints, hidden sizes for the edge message
        net, the last element in the list is the size of the message vectors
        [8, 64, 64] = [8, 64, 64]

        :param node_hidden_sizes: list of ints, hidden sizes for the node update
        net. [5, 32]

        :param node_update_type: type of node updates, one of {mlp, gru, residual}

        :param use_reverse_direction: set to True to also propagate messages in the
        reverse direction.

        :param reverse_dir_param_different: set to True to have the messages computed
        using a different set of parameters than for the forward direction

        :param layer_norm: set to True to use layer normalization in a few places
        '''

        super(GraphPropLayer, self).__init__()


        self.message_net = message_net
        self.reverse_message_net = reverse_message_net
        self.node_update_MLP = node_update_MLP


        self._node_state_dim = node_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]

        # output size is node_state_dim
        self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        #self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type

        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different

        self._layer_norm = layer_norm

        '''
        layers_sizes = [128] + self._edge_hidden_sizes #[node_state_dim*2 + edge_emb_dim (==config.node_state_dim*2)]
        self.message_net = MLP(layers_sizes) #MLP layers [128, 8, 64, 64]
        
        #self.reverse_message_net = MLP(layers_sizes)

        if self._node_update_type in ('mlp', 'residual'):
            mlp_layer_list = [128] + self._node_hidden_sizes #128 = edge_emb_dim + node_state_dim + additional_node_state_dim because node_states will be appended to the list
            # mlp_layer_list = [128, 5, 64, 32]

        self.node_mlp = MLP(mlp_layer_list)


        if torch.cuda.is_available() and config.cuda:
            self.message_net = self.message_net.cuda()
            #self.reverse_message_net = self.reverse_message_net.cuda()
            self.node_mlp = self.node_mlp.cuda()
        '''


    def _compute_aggregated_messages(self, node_states, from_idx, to_idx, edge_features=None):
        """Compute aggregated messages for each node.

        Args:
        :param node_states: [n_nodes, node_state_dim] float tensor, node states.
        [n_nodes, 32]; 32 because it has already gone through Graph encoder module,
        which has finsihed embedding the initial node features

        :param from_idx: [n_edges] int tensor, "from node" indices for each edge.

        :param to_idx: [n_edges] int tensor, "to node" indices for each edge.

        :param edge_features: if not None, should be [n_edges, edge_embedding_dim]
                 tensor, edge features.; [n_nodes, 64]; 64 because it has already
                 gone through Graph encoder module, which has finsihed embedding
                  the initial edge features


        Returns:
        aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
                            aggregated messages for each node.
        """


        aggregated_messages = graph_prop_once(node_states,
                                              from_idx,
                                              to_idx,
                                              self.message_net,
                                              #self._edge_hidden_sizes,
                                              aggregation_module=the_aggregation_module,
                                              edge_features=edge_features)




        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            print('you need to figure out a way to bridge the MLP inside this condition')

            if self._reverse_dir_param_different:
                reverse_message_net = self.reverse_message_net
            else:
                reverse_message_net = self.message_net

            reverse_aggregated_messages = graph_prop_once(node_states,
                                              to_idx,
                                              from_idx,
                                              reverse_message_net,
                                              #self._edge_hidden_size,
                                              aggregation_module=the_aggregation_module,
                                              edge_features=edge_features)

            aggregated_messages += reverse_aggregated_messages



        if self._layer_norm:
            aggregated_messages = nn.LayerNorm(aggregated_messages.size()[1:], elementwise_affine=False)(aggregated_messages)



        return aggregated_messages



    def _compute_node_update(self, node_states, node_state_inputs, node_features=None):
        """Compute node updates.

        Args:
        :param node_states: [n_nodes, node_state_dim] float tensor, the input node
                            states. [n_nodes, 32]

        :param node_state_inputs: a list of tensors used to compute node updates.  Each
        element tensor should have shape [n_nodes, feat_dim], where feat_dim can
        be different.  These tensors will be concatenated along the feature
        dimension.
         #for GMN, this is [aggregated_msg, attention]
         # aggregated msgs has the dim [n_nodes, edge_emb_dim], whereas
         attention has the dim [n_nodes, node_state_dim]


        :param node_features: extra node features if provided, should be of size
                [n_nodes, extra_node_feat_dim] float tensor, can be used to
                implement different types of skip connections.

        Returns:
        new_node_states: [n_nodes, node_state_dim] float tensor, the new node
        state tensor. [n_nodes, 32]

        Raises:
        ValueError: if node update type is not supported.
        """

        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            print('you need to add the GRU code here')
            exit()
            #_, new_node_states = snt.GRU(self._node_state_dim)(
            #node_state_inputs, node_states)
            #return new_node_states
        else:
            mlp_output = self.node_update_MLP(node_state_inputs)

        if self._layer_norm:
            mlp_output = nn.LayerNorm(mlp_output.size()[1:], elementwise_affine=False)(mlp_output)

        if self._node_update_type == 'mlp':
            return mlp_output
        elif self._node_update_type == 'residual':
            return node_states + mlp_output
        else:
            raise ValueError('Unknown node update type %s' % self._node_update_type)


    def forward(self, node_states, from_idx, to_idx, edge_features=None, node_features=None):
        """Run one propagation step.

        Args:
        :param node_states: [n_nodes, input_node_state_dim] float tensor, node states.

        :param from_idx: [n_edges] int tensor, from node indices for each edge.

        :param to_idx: [n_edges] int tensor, to node indices for each edge.

        :param edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.

        :param node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.
        """

        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        list_aggregated_msgs = [aggregated_messages]
        print(list_aggregated_msgs)

        return self._compute_node_update(node_states,
                                         [aggregated_messages],
                                         node_features=node_features)
