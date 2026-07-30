import numpy as np

import openturns as ot


def build_cluster_path(XS, w):
    """Build a greedy clustering path by repeatedly merging the smallest-weight
    point with its nearest neighbor."""
    workingXS = ot.Sample(XS)
    workingW = ot.Point(w)
    path = []
    while len(workingXS) > 1:
        i1 = int(np.argmin(workingW))
        X1 = workingXS[i1]
        w1 = workingW[i1]
        d_min = ot.SpecFunc.MaxScalar
        i2 = -1
        for j in range(len(workingXS)):
            if j == i1:
                continue
            d = (workingXS[j] - X1).normSquare()
            if d < d_min:
                d_min = d
                i2 = j
        X2 = workingXS[i2]
        w2 = workingW[i2]
        new_w = w1 + w2
        new_x = X1 * (w1 / new_w) + X2 * (w2 / new_w)
        path.append([i1, i2])
        workingXS[i1] = new_x
        workingW[i1] = new_w
        workingXS[i2] = workingXS[-1]
        workingW[i2] = workingW[-1]
        workingXS = workingXS[:-1]
        workingW = workingW[:-1]
    return path
