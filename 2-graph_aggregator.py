from message_passing_module_1 import *
import torch_scatter

AGGREGATION_TYPE = {
    'sum': torch_scatter.scatter_sum,
    'mean': torch_scatter.scatter_mean,
    'min': torch_scatter.scatter_min,
    'max': torch_scatter.scatter_max,
}

class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self, node_hidden_sizes, graph_transform_sizes=None,
               gated=True, aggregation_type='sum'):
        """Constructor.

        Args:
        :param node_hidden_sizes: the hidden layer sizes of the node transformation nets.
        The last element is the size of the aggregated graph representation.
        It is of the form: [node_feature_dim, node_state_dim]

        :param graph_transform_sizes: sizes of the transformation layers on top
        of the graph representations. The last element of this list is the final
        dimensionality of the output graph representations.

        :param gated: set to True to do gated aggregation, False not to.

        :param aggregation_type: one of {sum, max, mean, sqrt_n}.
        """

        super(GraphAggregator, self).__init__()

        self._gated = gated
        self._graph_state_dim = node_hidden_sizes[0]

        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes

        self._aggregation_type = aggregation_type
        self._aggregation_op = AGGREGATION_TYPE[aggregation_type]


        mlp_layer_sizes = [config.node_state_dim] + node_hidden_sizes #input to this mlp is [n_nodes, node_state_dim]
        self.node_state_g_mlp = MLP(mlp_layer_sizes) #mlp_layers: [64, 256]

        mlp_layer_sizes = [self._graph_transform_sizes[0], self._graph_transform_sizes[0]]
        # this mlp_layer_sizes is of the shape [config.graph_rep_dim, config.graph_rep_dim]
        self.graph_transform_mlp = MLP(mlp_layer_sizes) #mlp_layers: [256, 256]

        if torch.cuda.is_available() and config.cuda:
            self.node_state_g_mlp = self.node_state_g_mlp.cuda()
            self.graph_transform_mlp = self.graph_transform_mlp.cuda()

    def forward(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
        :param node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.

        :param graph_idx: [n_nodes] int tensor, graph ID for each node.

        :param n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """

        node_states_g = self.node_state_g_mlp(node_states)

        if self._gated:
            sigmoid = nn.Sigmoid()
            gates = sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        graph_states = self._aggregation_op(node_states_g, graph_idx.long(), dim=0)
        # graph_states: [n_graphs, self._graph_state_dim] = [n_graphs, 128]

        # pylint: disable=g-explicit-length-test
        if (self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0):
            graph_states = self.graph_transform_mlp(graph_states)

        return graph_states
