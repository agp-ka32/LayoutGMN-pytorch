from cross_graph_attention_4 import *

class GraphPropMatchingLayer(GraphPropLayer):
    """"A graph propagation layer that also does cross graph matching.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """
    #def __init__(self):
        #super(GraphPropMatchingLayer, self).__init__()


    def forward(self,
             node_states,
             from_idx,
             to_idx,
             graph_idx,
             n_graphs,
             similarity='dotproduct',
             edge_features=None,
             node_features=None):
        """Run one propagation step with cross-graph matching.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          graph_idx: [n_onodes] int tensor, graph id for each node.
          n_graphs: integer, number of graphs in the batch.
          similarity: type of similarity to use for the cross graph attention.
          edge_features: if not None, should be [n_edges, edge_feat_dim] tensor,
            extra edge features.
          node_features: if not None, should be [n_nodes, node_feat_dim] tensor,
            extra node features.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.

        Raises:
          ValueError: if some options are not provided correctly.
        """
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        # new stuff here
        cross_graph_attention = batch_block_pair_attention(
            node_states, graph_idx, n_graphs, similarity=similarity)

        attention_input = node_states - cross_graph_attention
        #attention_input output size is[n_nodes, node_state_dim], i.e., [n_nodes, 32]
        return self._compute_node_update(node_states,
                                         [aggregated_messages, attention_input],
                                         node_features=node_features)


class GraphMatchingNet(GraphEmbeddingNet):
    """Graph matching net.

    This class uses graph matching layers instead of the simple graph prop layers.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
               encoder,
               aggregator,
               message_net,
               reverse_message_net,
               node_update_MLP,
               node_state_dim,
               edge_hidden_sizes,
               node_hidden_sizes,
               n_prop_layers,
               share_prop_params=False,
               #edge_net_init_scale=0.1,
               node_update_type='residual',
               use_reverse_direction=True,
               reverse_dir_param_different=True,
               layer_norm=False,
               similarity='dotproduct'):
        super(GraphMatchingNet, self).__init__(
            encoder,
            aggregator,
            message_net,
            reverse_message_net,
            node_update_MLP,
            node_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            n_prop_layers,
            share_prop_params=share_prop_params,
            node_update_type=node_update_type,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm)
        self._similarity = similarity
        self._layer_class = GraphPropMatchingLayer


    def _apply_gmn_layer(self,
                   layer,
                   node_states,
                   from_idx,
                   to_idx,
                   graph_idx,
                   n_graphs,
                   edge_features):
        """Apply one layer on the given inputs."""
        return layer(node_states, from_idx, to_idx, graph_idx, n_graphs,
                 similarity=self._similarity, edge_features=edge_features)


def euclidean_distance(x, y):
  """This is the squared Euclidean distance."""
  return torch.sum((x - y)**2, dim=-1)


def approximate_hamming_similarity(x, y):
  """Approximate Hamming similarity."""
  return torch.mean(F.tanh(x) * F.tanh(y), dim=1)



def triplet_loss(x_1, y, x_2, z, loss_type='margin', margin=1.0):
    """Compute triplet loss.

    This function computes loss on a triplet of inputs (x, y, z).  A similarity or
    distance value is computed for each pair of (x, y) and (x, z).  Since the
    representations for x can be different in the two pairs (like our matching
    model) we distinguish the two x representations by x_1 and x_2.

    Args:
        x_1: [N, D] float tensor.
        y: [N, D] float tensor.
        x_2: [N, D] float tensor.
        z: [N, D] float tensor.
        loss_type: margin or hamming.
        margin: float scalar, margin for the margin loss.

    Returns:
        loss: [N] float tensor.  Loss for each pair of representations.
    """
    if loss_type == 'margin':
        return F.relu(margin +
                      euclidean_distance(x_1, y) -
                      euclidean_distance(x_2, z))
    elif loss_type == 'hamming':
        return 0.125 * ((approximate_hamming_similarity(x_1, y) - 1)**2 +
                    (approximate_hamming_similarity(x_2, z) + 1)**2)
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)
