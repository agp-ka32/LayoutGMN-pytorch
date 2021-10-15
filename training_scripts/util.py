from argparse import ArgumentParser

def get_args():
    """ The default configs"""
    parser = ArgumentParser(description='Graph Matching Network on Layout Data')

    parser.add_argument('--train_mode', action='store_true', default=False)
    parser.add_argument('--eval_mode', action='store_true', default=True)

    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--graph_vec_regularizer_weight', type=float,
                        default=1e-6, help='this is to avoid graph vecs blowing up')
    parser.add_argument('--clip_val', type=float, default=10.0,
                        help='gradient clipping to avoid large gradients')

    parser.add_argument('--epochs', type=int, default=700)
    #parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--data_mode', type=str, default='triplet',
                        help='whether to use paired data or triplet data')
    parser.add_argument('--loss_type', type=str, default='margin',
                        help='whether or not to use margin loss')
    parser.add_argument('--margin_val', type=float, default=1, help='margin value')
    parser.add_argument('--no_plot', action='store_true', default=False)
    parser.add_argument('--show_log_every', type=int, default=5)
    parser.add_argument('--save_network_every', type=int, default=10)
    parser.add_argument('--load_pretrained', action='store_true', default=True)


    parser.add_argument('--use_box_feats', action='store_true', default=True,
                        help='whether to use geometric box features')
    parser.add_argument('--apn_dict_path', type=str, default='/gruvi/usr/akshay/1-FPs/7-FP_Metric/GCN_CNN_data/data/Triplets/apn_dict_13K_pthres60.pkl',
                        help='path to the training triplets computed based on IoU')
    parser.add_argument('--hardmining', action='store_true', default=False,
                        help='whether to use geometric box features')

    parser.add_argument('--node_geometry_feat_dim', type=int, default=5,
                        help='Initial node features for the graph without room semantics -- this is before doing any forward pass')
    parser.add_argument('--edge_feat_dim', type=int, default=8,
                        help='Initial edge features for the graph -- this is before doing any forward pass')
    parser.add_argument('--node_state_dim', type=int, default=128,
                        help='Embedding dimension of the nodes after message passing')
    parser.add_argument('--graph_rep_dim', type=int, default=1024,
                        help='Embedding dimension of the entire graph')
    parser.add_argument('--n_prop_layers', type=int, default=5,
                        help='Number of propagation steps')
    parser.add_argument('--share_prop_params', action='store_true', default=False,
                        help='Whether to share propagation parameters in the reverse direction')

    parser.add_argument('--node_update_type', type=str, default='residual',
                        help='One of the three -- GRU, MLP or Residual')
    parser.add_argument('--use_reverse_direction', action='store_true', default=False,
                        help='whether to send messages in the reverse direction along'
                             ' an edge; set to False if graph already contains edges '
                             'in both dirs')
    parser.add_argument('--reverse_dir_param_different', action='store_true',
                        default=True, help='Set this to True if your graph is directed')
    parser.add_argument('--layer_norm', action='store_true', default=True,
                        help='normalizing aggregated messages after one layer of prop')
    parser.add_argument('--model_type', type=str, default='matching',
                        help='Whether to use GNN or GMN')

    parser.add_argument('--model_save_path', default='trained_models/')


    args = parser.parse_args()
    return args