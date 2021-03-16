from distance_metrics import *


def compute_cross_attention(x, y, sim):
    """Compute cross attention.

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    Args:
        x: NxD float tensor.
        y: MxD float tensor.
        sim: a (x, y) -> similarity function.

    Returns:
        attention_x: NxD float tensor.
        attention_y: NxD float tensor.
    """
    a = sim(x, y)
    a_x = nn.Softmax(dim=1)(a)  # i->j
    a_y = nn.Softmax(dim=0)(a) # j->i
    attention_x = torch.matmul(a_x, y)

    ay_transpose = torch.transpose(a_y, 0, 1)
    attention_y = torch.matmul(ay_transpose, x)#x_transpose)
    return attention_x, attention_y


def torch_dynamic_partition(data, partitions, num_partitions):
    res = []
    for i in range(num_partitions):
        res += [data[(partitions == i).nonzero().squeeze(1)]]
    return res



def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,
                               similarity='dotproduct'):
    """Compute batched attention between pairs of blocks.

    This function partitions the batch data into blocks according to block_idx.
    For each pair of blocks, x = data[block_idx == 2i], and
    y = data[block_idx == 2i+1], we compute

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    and

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.

    Args:
        data: NxD float tensor.
        block_idx: N-dim int tensor.
        n_blocks: integer.
        similarity: a string, name of the similarity metric.

    Returns:
        attention_output: NxD float tensor, each x_i replaced by attention_x_i.

    Raises:
        ValueError: if n_blocks is not an integer or not a multiple of 2.
    """
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

    sim = get_pairwise_similarity(similarity)

    results = []

    # This is probably better than doing boolean_mask for each i
    partitions = torch_dynamic_partition(data, block_idx, n_blocks)

    # It is rather complicated to allow n_blocks be a tf tensor and do this in a
    # dynamic loop, and probably unnecessary to do so.  Therefore we are
    # restricting n_blocks to be a integer constant here and using the plain for
    # loop.
    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)

    results = torch.cat(results, dim=0)

    # the shape of the first dimension is lost after concat, reset it back
    #results.set_shape(data.shape)

    assert results.shape == data.shape
    return results
