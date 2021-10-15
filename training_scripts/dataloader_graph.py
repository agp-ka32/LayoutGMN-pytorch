import torch
import numpy as np

from ged_graph_data import flatten_nested_tuples


class data_input_to_gmn(object):
    def __init__(self, config, device, batch_sg_a, batch_sg_p, batch_sg_n):
    #def __init__(self, batch_sg_a, batch_sg_p, batch_sg_n):
        '''
        :param config: config file used for training
        :param device: the device (gpu o, 1, or multiple) on which to port the models
        :param batch_sg_a: batch grapphs of the anchors
        :param batch_sg_p: batch graphs for positive examples
        :param batch_sg_n: batch graphs for negative examples

        '''
        super(data_input_to_gmn, self).__init__()
        self.config = config
        self.device = device
        self.batch_sg_a = batch_sg_a
        self.batch_sg_p = batch_sg_p
        self.batch_sg_n = batch_sg_n

    def quadruples(self):
        '''
        :return (the overall class: a tuple of quadruples (g_a, g_p, g_a, g_n)
        '''

        assert len(self.batch_sg_a) == len(self.batch_sg_p) == len(self.batch_sg_n)
        batch_quadruples = []

        for i in range(len(self.batch_sg_a)):
            batch_quadruples.append((self.batch_sg_a[i], self.batch_sg_p[i],
                                    self.batch_sg_a[i], self.batch_sg_n[i]))

        batch_quadruples = tuple(batch_quadruples)
        data = self._pack_batch(batch_quadruples)
        return data

    def _pack_batch(self, batch_quadruples):
        """Pack a batch of "batch_quadruples" into a single `GraphData` instance.

        Args:
          graphs: a tuple of quadruples.

        Returns:
          graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """

        graphs = flatten_nested_tuples(batch_quadruples)
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        node_geometry_feats = []
        node_room_ids = []
        edge_feats = []
        for i, g in enumerate(graphs):
            n_nodes = g['box_feats'].shape[0]
            n_edges = g['rela_edges'].shape[0]

            node_geometry_feats.append(torch.from_numpy(g['box_feats']).squeeze(0))
            node_room_ids.append(torch.from_numpy(g['room_ids']).squeeze(0))
            edge_feats.append(torch.from_numpy(g['rela_feats']).squeeze(0))

            edges = np.array(g['rela_edges'], dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        node_geometry_feats = torch.cat(node_geometry_feats, dim=0).float()
        node_room_ids = torch.cat(node_room_ids, dim=0).float()
        edge_feats = torch.cat(edge_feats, dim=0).float()


        #'''
        if self.config.cuda: #on GPU
            return {'node_geometry_features': node_geometry_feats.to(self.device), #torch.from_numpy(np.ones((n_total_nodes, 5), dtype=np.float32)),
                    'node_room_ids': node_room_ids.to(self.device),
                    'edge_features': edge_feats.to(self.device), #torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                    'from_idx': torch.from_numpy(np.concatenate(from_idx, axis=0)).long().to(self.device),
                    'to_idx': torch.from_numpy(np.concatenate(to_idx, axis=0)).long().to(self.device),
                    'graph_idx': torch.from_numpy(np.concatenate(graph_idx, axis=0)).to(self.device),#.long(),
                    'n_graphs': len(graphs)
                    }
        else: #on CPU
            return {'node_geometry_features': node_geometry_feats.to(self.device), #torch.from_numpy(np.ones((n_total_nodes, 5), dtype=np.float32)),
                    'node_room_ids': node_room_ids.to(self.device),
                    'edge_features': edge_feats.to(self.device), #torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                    # torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                    'from_idx': torch.from_numpy(np.concatenate(from_idx, axis=0)).long(),#.to(self.device),
                    'to_idx': torch.from_numpy(np.concatenate(to_idx, axis=0)).long(),#.to(self.device),
                    'graph_idx': torch.from_numpy(np.concatenate(graph_idx, axis=0)),#.to(self.device),  # .long(),
                    'n_graphs': len(graphs)
                    }

        #'''


        '''
        return {'node_features': node_feats,#.to(self.device),  # torch.from_numpy(np.ones((n_total_nodes, 5), dtype=np.float32)),
                    'edge_features': edge_feats,#.to(self.device),
                    # torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                    'from_idx': torch.from_numpy(np.concatenate(from_idx, axis=0)).long(),#.to(self.device),
                    'to_idx': torch.from_numpy(np.concatenate(to_idx, axis=0)).long(),#.to(self.device),
                    'graph_idx': torch.from_numpy(np.concatenate(graph_idx, axis=0)),#.to(self.device),  # .long(),
                    'n_graphs': len(graphs)
                    }
        '''
        
