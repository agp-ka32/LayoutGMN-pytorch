from ged_graph_data import *
from util import get_args

config = get_args()

encoder_model =  GraphEncoder(config, node_hidden_sizes=[config.node_geometry_feat_dim, config.node_state_dim],
                              edge_hidden_sizes=[config.edge_feat_dim, int(config.node_state_dim)])

aggregator_model = GraphAggregator(node_hidden_sizes=[config.graph_rep_dim],
          graph_transform_sizes=[config.graph_rep_dim],
          gated=True,
          aggregation_type='sum')


message_net = MLP([2*config.node_state_dim+int(config.node_state_dim), config.edge_feat_dim, int(config.node_state_dim), int(config.node_state_dim)])
reverse_message_net = MLP([2*config.node_state_dim+int(config.node_state_dim), config.edge_feat_dim, int(config.node_state_dim), int(config.node_state_dim)])
node_update_mlp = MLP([2*config.node_state_dim+int(config.node_state_dim), config.node_geometry_feat_dim, int(config.node_state_dim), config.node_state_dim])


gmn_net = GraphMatchingNet(encoder = encoder_model,
               aggregator = aggregator_model,
               message_net = message_net,
               reverse_message_net = reverse_message_net,
               node_update_MLP = node_update_mlp,
               node_state_dim = config.node_state_dim,
               edge_hidden_sizes = [config.edge_feat_dim, config.node_state_dim * 2,
                                    config.node_state_dim * 2],
               node_hidden_sizes = [config.node_geometry_feat_dim, config.node_state_dim * 2],
               n_prop_layers = 5,
               share_prop_params=False,
               #edge_net_init_scale=0.1,
               node_update_type='residual',
               use_reverse_direction=False,
               reverse_dir_param_different=False,
               layer_norm=False,
               similarity='dotproduct')


def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
        tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
        n_splits: int, number of splits to split the tensor into.

    Returns:
        splits: a list of `n_splits` tensors.
        The first split is [tensor[0],tensor[n_splits], tensor[n_splits * 2], ...],
        the second split is [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """

    feature_dim = tensor.shape[-1]
    # feature dim must be known, otherwise you can provide that as an input
    assert isinstance(feature_dim, int)
    tensor = tensor.reshape([-1, feature_dim * n_splits])
    tuple = torch.split(tensor, feature_dim, dim=-1)

    return tuple[0], tuple[1], tuple[2], tuple[3]



def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = torch.eq(x > 0, y > 0).float()
    return torch.mean(match, dim=1)


def compute_similarity(config, x, y):
      """Compute the distance between x and y vectors.

      The distance will be computed based on the training loss type.

      Args:
        config: ArgParse arguments.
        x: [n_examples, feature_dim] float tensor.
        y: [n_examples, feature_dim] float tensor.

      Returns:
        dist: [n_examples] float tensor.

      Raises:
        ValueError: if loss type is not supported.
      """
      if config.loss_type == 'margin':
        # similarity is negative distance
        return -euclidean_distance(x, y)
      elif config.loss_type == 'hamming':
        return exact_hamming_similarity(x, y)
      else:
        raise ValueError('Unknown loss type %s' % config.loss_type)


# Before uncommenting and running this script, uncomment the "import" lines
# at the beginning of this script


if __name__ == '__main__':
    gmn_model = gmn_net
    gmn_model_params = list(gmn_model.parameters())
    GraphData = graph_data((4,8), (0.2, 1), 1, 2).triplets(1)

    graph_vectors = gmn_model(**GraphData)
    #print(graph_vectors)
    x1, y, x2, z = reshape_and_split_tensor(graph_vectors, 4)
    loss = triplet_loss(x1, y, x2, z, loss_type=config.loss_type, margin=config.margin_val)
    sim_pos = torch.mean(compute_similarity(config, x1, y))
    sim_neg = torch.mean(compute_similarity(config, x2, y))

    print(sim_pos)
    print(sim_neg)


    graph_vec_scale = torch.mean(graph_vectors**2)
    if config.graph_vec_regularizer_weight > 0:
        loss += config.graph_vec_regularizer_weight * 0.5 * graph_vec_scale
