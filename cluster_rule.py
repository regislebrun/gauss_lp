import openturns as ot


def cluster_rule(XS, w, path, target_N):
    """Apply a precomputed clustering path to obtain target_N clusters.

    Each merge replaces two points by their weighted convex combination.
    """
    workingXS = ot.Sample(XS)
    workingW = ot.Point(w)
    for idx in range(len(path)):
        if len(workingXS) <= target_N:
            break
        i1, i2 = path[idx]
        X1 = workingXS[i1]
        w1 = workingW[i1]
        X2 = workingXS[i2]
        w2 = workingW[i2]
        new_w = w1 + w2
        new_x = X1 * (w1 / new_w) + X2 * (w2 / new_w)
        workingXS[i1] = new_x
        workingW[i1] = new_w
        workingXS[i2] = workingXS[-1]
        workingW[i2] = workingW[-1]
        workingXS = workingXS[:-1]
        workingW = workingW[:-1]
    return workingXS, workingW
