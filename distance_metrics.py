from gnn_3 import *
#from ged_graph_data import *

def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

    Args:
        x: NxD float tensor.
        y: MxD float tensor.

    Returns:
        s: NxM float tensor, the pairwise euclidean similarity.
    """
    y_transpose = torch.transpose(y, 0, 1)
    s = torch.matmul(x, y_transpose)

    diag_x = torch.sum(x * x, dim=-1, keepdim=True)#.unsqueeze(-1)
    diag_y = torch.sum(y * y, dim=-1).reshape([1,-1])

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i * y_j^T.

    Args:
        x: NxD float tensor.
        y: MxD float tensor.

    Returns:
        s: NxM float tensor, the pairwise dot product similarity.
    """
    y_transpose = torch.transpose(y, 0, 1)
    return torch.matmul(x, y_transpose)


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
        x: NxD float tensor.
        y: MxD float tensor.

    Returns:
        s: NxM float tensor, the pairwise cosine similarity.
    """
    x = torch.norm(x, dim=-1)
    y = torch.norm(y, dim=-1)

    y_transpose = torch.transpose(y, 0, 1)
    return torch.matmul(x, y_transpose)


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.

    Args:
    :param name: string, name of the similarity metric, one of {dot-product, cosine,
        euclidean}.

    Returns:
        similarity: a (x, y) -> sim function.

    Raises:
        ValueError: if name is not supported.
    """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]