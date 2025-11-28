import numpy as np
import torch


def average(gradients):
    """ Aggregate the gradients using the average aggregation rule."""
    # Assertion

    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    if len(gradients) > 1:
        return torch.mean(gradients, 0)
    else:
        return gradients[0]

def sparsified_average_old(gradients):
    if len(gradients) > 1:
        # Mask zeros and compute the sum and count of non-zero elements
        non_zero_mask = gradients != 0.0
        sums = torch.sum(gradients, dim=0)
        counts = torch.sum(non_zero_mask, dim=0)
        # Compute mean safely, avoiding division by zero
        mean_tensor = torch.where(counts > 0, sums / counts, torch.tensor(0.0))
        return mean_tensor
    else:
        return gradients[0]

import torch

def sparsified_average(weights: torch.Tensor) -> torch.Tensor:
    """Returns average after “filling in” each neighbor’s missing coords with your own values."""
    # 1) If no neighbors, just return your model
    if weights.size(0) == 1:
        return weights[0]

    # 2) Broadcast your dense model to match shape [n_peers, …]
    local = weights[0]                              # shape […]
    local_expanded = local.unsqueeze(0)             # shape [1, …]

    # 3) Reconstruct each neighbor's full model by replacing zeros
    #    with your own parameter at that position
    full_models = torch.where(
        weights != 0.0,     # where neighbor has sent a value
        weights,            #   keep it
        local_expanded      # else fall back to local
    )                      # shape [n_peers, …]

    # 4) Now do a plain element-wise average across all peers
    return full_models.mean(dim=0)  # shape […]


def median(gradients):
    """ Aggregate the gradients using the median aggregation rule."""
    exit("Median uses numpy")
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    return np.median(gradients, 0)


def aksel(gradients):
    """ Aggregate the gradients using the AKSEL aggregation rule."""
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    med = np.median(gradients, axis=0)
    matrix = gradients - med
    normsq = [np.linalg.norm(grad) ** 2 for grad in matrix]
    med_norm = np.median(normsq)
    correct = [gradients[i] for i, norm in enumerate(normsq) if norm <= med_norm]

    return np.mean(correct, axis=0)


def krum(gradients, f=1):
    """ Aggregate the gradients using the Krum aggregation rule."""
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    nbworkers = len(gradients)
    gradients = np.array(gradients)
    # Distance computations
    scores = []
    sqr_dst = []
    for i in range(nbworkers - 1):
        sqr_dst = []
        gi = gradients[i].reshape(-1, 1)
        for j in range(nbworkers - 1):
            gj = gradients[j].reshape(-1, 1)
            dst = np.linalg.norm(gi - gj) ** 2
            sqr_dst.append(dst)
        indices = list(np.argsort(sqr_dst)[:nbworkers - f - 2])
        sqr_dst = np.array(sqr_dst)
        scores.append(np.sum(sqr_dst[indices]))
    correct = np.argmin(scores)

    return gradients[correct]


if __name__ == '__main__':
    # g = [[1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3]]
    g = [[1., 3., 4.], [9., 7., 6.], [5., 5., 5.]]
    tg = torch.tensor(g)
    # print(median(tg))
    print(np.mean(tg, 0))
