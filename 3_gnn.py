from graph_aggregator_2 import *

class GraphEmbeddingNet(nn.Module):
    """A graph to embedding mapping network."""

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
                   node_update_type='residual',
                   use_reverse_direction=True,
                   reverse_dir_param_different=True,
                   layer_norm=False):
        """Constructor.

        Args:
          :param encoder: GraphEncoder, encoder that maps initial features to embeddings.

          :param aggregator: GraphAggregator, aggregator that produces graph
            representations.

          :param node_state_dim: dimensionality of node states.

          :param edge_hidden_sizes: sizes of the hidden layers of the edge message nets.

          :param node_hidden_sizes: sizes of the hidden layers of the node update nets.

          :param n_prop_layers: number of graph propagation layers.

          :param share_prop_params: set to True to share propagation parameters across all
            graph propagation layers, False not to.

          :param edge_net_init_scale: scale of initialization for the edge message nets.

          :param node_update_type: type of node updates, one of {mlp, gru, residual}.

          :param use_reverse_direction: set to True to also propagate messages in the
            reverse direction.

          :param reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.

          :param layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphEmbeddingNet, self).__init__()

        self._encoder = encoder
        self._aggregator = aggregator
        self.message_net = message_net
        self.reverse_message_net = reverse_message_net
        self.node_update_mlp = node_update_MLP
        self._node_state_dim = node_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        self._n_prop_layers = n_prop_layers
        self._share_prop_params = share_prop_params
        #self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._layer_norm = layer_norm

        self._prop_layers = []
        self._layer_class = GraphPropLayer


    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            self.message_net,
            self.reverse_message_net,
            self.node_update_mlp,
            self._node_state_dim,
            self._edge_hidden_sizes,
            self._node_hidden_sizes,
            node_update_type=self._node_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm)

    def _apply_gmn_layer(self,
                       layer,
                       node_states,
                       from_idx,
                       to_idx,
                       graph_idx,
                       n_graphs,
                       edge_features):
        """Apply one layer on the given inputs."""
        del graph_idx, n_graphs
        return layer(node_states, from_idx, to_idx, edge_features=edge_features)


    #def _build(self...):
    def forward(self,
                 node_geometry_features,
                 node_room_ids,
                 edge_features,
                 from_idx,
                 to_idx,
                 graph_idx,
                 n_graphs):
        """Compute graph representations.

        Args:
          :param node_features: [n_nodes, node_feat_dim] float tensor.

          :param edge_features: [n_edges, edge_feat_dim] float tensor.

          :param from_idx: [n_edges] int tensor, index of the from node for each edge.

          :param to_idx: [n_edges] int tensor, index of the to node for each edge.

          :param graph_idx: [n_nodes] int tensor, graph id for each node.

          :param n_graphs: int, number of graphs in the batch.

        Returns:
          graph_representations: [n_graphs, graph_representation_dim] float tensor,
            graph representations.
        """
        if len(self._prop_layers) < self._n_prop_layers:
          # build the layers
          for i in range(self._n_prop_layers):
            if i == 0 or not self._share_prop_params:
              layer = self._build_layer(i)
            else:
              layer = self._prop_layers[0]
            self._prop_layers.append(layer)

        node_features, edge_features = self._encoder(node_geometry_features, node_room_ids, edge_features)
        node_states = node_features

        layer_outputs = [node_states]

        for layer in self._prop_layers:
          # node_features could be wired in here as well, leaving it out for now as
          # it is already in the inputs
          node_states = self._apply_gmn_layer(
              layer,
              node_states,
              from_idx,
              to_idx,
              graph_idx,
              n_graphs,
              edge_features)
          layer_outputs.append(node_states)

        # these tensors may be used e.g. for visualization
        self._layer_outputs = layer_outputs
        return self._aggregator(node_states, graph_idx, n_graphs)


    def reset_n_prop_layers(self, n_prop_layers):
        """Set n_prop_layers to the provided new value.

        This allows us to train with certain number of propagation layers and
        evaluate with a different number of propagation layers.

        This only works if n_prop_layers is smaller than the number used for
        training, or when share_prop_params is set to True, in which case this can
        be arbitrarily large.

        Args:
          n_prop_layers: the new number of propagation layers to set.
        """
        self._n_prop_layers = n_prop_layers

    @property
    def n_prop_layers(self):
        return self._n_prop_layers

    def get_layer_outputs(self):
        """Get the outputs at each layer."""
        if hasattr(self, '_layer_outputs'):
          return self._layer_outputs
        else:
          raise ValueError('No layer outputs available.')
