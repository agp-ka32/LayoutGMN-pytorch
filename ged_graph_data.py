import networkx as nx
import copy
import numpy as np

from cross_graph_message_passing_5 import *
import collections
import abc
import six

'''
GraphData = collections.namedtuple('GraphData', [
    'node_features',
    'edge_features',
    'from_idx',
    'to_idx',
    'graph_idx',
    'n_graphs'])
'''


@six.add_metaclass(abc.ABCMeta)
class GraphSimilarityDataset(object):
    """Base class for all the graph similarity learning datasets.

    This class defines some common interfaces a graph similarity dataset can have,
    in particular the functions that creates iterators over pairs and triplets.
    """

    @abc.abstractmethod
    def triplets(self, batch_size):
        """Create an iterator over triplets.

        Args:
            batch_size: int, number of triplets in a batch.

        Yields:
            graphs: a `GraphData` instance.  The batch of triplets put together.  Each
            triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
            so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
            The batch contains `batch_size` number of triplets, hence `4*batch_size`
            many graphs.
        """
        pass



def flatten_nested_tuples(nested_tuple):
    '''

    :param nested_tuple: a tuple of tuples (maybe int or other data structure)
            Ex: ((1,2,3), (4,5,6), (7,8,9), (10,11,12))
    :return: a list, with entries being the nested-tuple entires, in the same order
    '''
    flattened_list = []

    for sub_tuple in nested_tuple: #sub_tuple is something like (1,2,3)
        for var in sub_tuple:
            flattened_list.append(var)

    return flattened_list



#class graph_data(object):
class graph_data(GraphSimilarityDataset):
    """Graph edit distance dataset."""

    def __init__(self, n_nodes_range, p_edge_range, n_changes_positive, n_changes_negative, permute=True):
        """Constructor.

        Args:
            :param n_nodes_range: a tuple (n_min, n_max).  The minimum and maximum number of
                            nodes in a graph to generate.

            :param p_edge_range: a tuple (p_min, p_max).  The minimum and maximum edge
                            probability.

            :param n_changes_positive: the number of edge substitutions for a pair to be
                            considered positive (similar).

            :param n_changes_negative: the number of edge substitutions for a pair to be
                            considered negative (not similar).

            :param permute: if True (default), permute node orderings in addition to
                    changing edges; if False, the node orderings across a pair or triplet of
                    graphs will be the same, useful for visualization.
        """
        #super(graph_data, self).__init__()
        self._n_min, self._n_max = n_nodes_range
        self._p_min, self._p_max = p_edge_range
        self._k_pos = n_changes_positive
        self._k_neg = n_changes_negative
        self._permute = permute


    def permute_graph_nodes(self, g):
        """Permute node ordering of a graph, returns a new graph."""
        n = g.number_of_nodes()
        new_g = nx.Graph()
        new_g.add_nodes_from(range(n))
        perm = np.random.permutation(n)
        edges = g.edges()
        new_edges = []
        for x, y in edges:
            new_edges.append((perm[x], perm[y]))
        new_g.add_edges_from(new_edges)
        return new_g

    def substitute_random_edges(self, g, n):
        """Substitutes n edges from graph g with another n randomly picked edges."""
        g = copy.deepcopy(g)
        n_nodes = g.number_of_nodes()
        edges = list(g.edges())
        # sample n edges without replacement
        e_remove = [edges[i] for i in np.random.choice(
            np.arange(len(edges)), n, replace=False)]
        edge_set = set(edges)
        e_add = set()

        while len(e_add) < n:
            e = np.random.choice(n_nodes, 2, replace=False)
            # make sure e does not exist and is not already chosen to be added
            if ((e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set and
                        (e[0], e[1]) not in e_add and (e[1], e[0]) not in e_add):
                e_add.add((e[0], e[1]))

        for i, j in e_remove:
            g.remove_edge(i, j)
        for i, j in e_add:
            g.add_edge(i, j)
        return g

    def _get_graph(self):
        """Generate one graph."""
        n_nodes = np.random.randint(self._n_min, self._n_max + 1)
        p_edge = np.random.uniform(self._p_min, self._p_max)

        # do a little bit of filtering
        n_trials = 100
        for _ in range(n_trials):
            g = nx.erdos_renyi_graph(n_nodes, p_edge)
            if nx.is_connected(g):
                return g

        raise ValueError('Failed to generate a connected graph.')


    def _get_triplet(self):
        """Generate one triplet of graphs."""
        g = self._get_graph()
        if self._permute:
            permuted_g = self.permute_graph_nodes(g)
        else:
            permuted_g = g
        pos_g = self.substitute_random_edges(g, self._k_pos)
        neg_g = self.substitute_random_edges(g, self._k_neg)
        return permuted_g, pos_g, neg_g



    def triplets(self, batch_size):
        """Yields batches of triplet data."""
        #while True:
        batch_graphs = []
        for _ in range(batch_size):
            g1, g2, g3 = self._get_triplet()
            batch_graphs.append((g1, g2, g1, g3))
        data = self._pack_batch(batch_graphs)
        return data


    def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.

        Args:
          graphs: a list of generated networkx graphs.

        Returns:
          graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """
        #graphs = tf.nest.flatten(graphs)
        graphs = flatten_nested_tuples(graphs)
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges
        '''
        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features
            node_features=np.ones((n_total_nodes, 1), dtype=np.float32),
            edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs))
        '''
        return {'node_features': torch.from_numpy(np.ones((n_total_nodes, 17), dtype=np.float32)),
                'edge_features':torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)),
                'from_idx': torch.from_numpy(np.concatenate(from_idx, axis=0)).long(),
                'to_idx': torch.from_numpy(np.concatenate(to_idx, axis=0)).long(),
                'graph_idx' : torch.from_numpy(np.concatenate(graph_idx, axis=0)),
                'n_graphs' : len(graphs)
                }

'''
if __name__ == '__main__':
    #nested_tuple = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))
    #fllattened_list = flatten_nested_tuples(nested_tuple)
    #print(fllattened_list)

    #graph_sim_dataset = GraphSimilarityDataset()
    #my_obj = GraphEditDistanceDataset(graph_sim_dataset)
    my_obj = graph_data((3,6), (0.2, 1), 1, 2)
    GraphData = my_obj.triplets(1)
    #x = **GraphData
    print(GraphData)
'''











































#class GraphEditDistanceDataset(GraphSimilarityDataset):
class GraphDataset(object):
    """Graph edit distance dataset."""

    def __init__(self, batched_graphs):

        '''

        :param batched_graphs: a list of tuples of the form
                                [(g11, g12, g11 , g13),
                                 (g21, g22, g21, g23),
                                 (g31, g32, g31, g33),
                                 (g41, g42, g41, g43)]

        return: 'GraphData' tuple
        '''
        super(GraphDataset, self).__init__()
        self.batched_graphs = batched_graphs
        yield self._pack_batch(self.batched_graphs)


    def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.

        Args:
          graphs: a list of generated networkx graphs.

        Returns:
          graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """
        #graphs = tf.nest.flatten(graphs)
        graphs = flatten_nested_tuples(graphs)
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features
            node_features=np.ones((n_total_nodes, 1), dtype=np.float32),
            edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs))



'''
@contextlib.contextmanager
def reset_random_state(seed):
    """This function creates a context that uses the given seed."""
    np_rnd_state = np.random.get_state()
    rnd_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed + 1)
    try:
        yield
    finally:
        random.setstate(rnd_state)
        np.random.set_state(np_rnd_state)



class FixedGraphEditDistanceDataset(GraphEditDistanceDataset):
    """A fixed dataset of pairs or triplets for the graph edit distance task.

    This dataset can be used for evaluation.
    """

    def __init__(self,
                   n_nodes_range,
                   p_edge_range,
                   n_changes_positive,
                   n_changes_negative,
                   dataset_size,
                   permute=True,
                   seed=1234):
        super(FixedGraphEditDistanceDataset, self).__init__(
            n_nodes_range, p_edge_range, n_changes_positive, n_changes_negative,
            permute=permute)
        self._dataset_size = dataset_size
        self._seed = seed



    def triplets(self, batch_size):
        """Yield triplets."""

        if hasattr(self, '_triplets'):
            triplets = self._triplets
        else:
            # get a fixed set of triplets
            with reset_random_state(self._seed):
                triplets = []
                for _ in range(self._dataset_size):
                    g1, g2, g3 = self._get_triplet()
                    triplets.append((g1, g2, g1, g3))
            self._triplets = triplets

        ptr = 0
        while ptr + batch_size <= len(triplets):
            batch_graphs = triplets[ptr:ptr + batch_size]
            yield self._pack_batch(batch_graphs)
            ptr += batch_size



    def pairs(self, batch_size):
        """Yield pairs and labels."""

        if hasattr(self, '_pairs') and hasattr(self, '_labels'):
            pairs = self._pairs
            labels = self._labels
        else:
            # get a fixed set of pairs first
            with reset_random_state(self._seed):
                pairs = []
                labels = []
                positive = True
                for _ in range(self._dataset_size):
                  pairs.append(self._get_pair(positive))
                  labels.append(1 if positive else -1)
                  positive = not positive
            labels = np.array(labels, dtype=np.int32)

            self._pairs = pairs
            self._labels = labels

        ptr = 0
        while ptr + batch_size <= len(pairs):
            batch_graphs = pairs[ptr:ptr + batch_size]
            packed_batch = self._pack_batch(batch_graphs)
            yield packed_batch, labels[ptr:ptr + batch_size]
            ptr += batch_size
'''