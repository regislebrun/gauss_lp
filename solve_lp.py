import openturns as ot
import openturns.experimental as otexp


def solve_lp(XS, polynoms, moments, sensitivity_values):
    """Solve the discretized LP (eq. 3 in Ryu 2013).

    Returns the vector of weights (alpha_i >= 0) for each sample point.
    """
    S = len(XS)
    n = len(polynoms)
    A = ot.DesignProxy(XS, polynoms).computeDesign(list(range(n))).transpose()
    constraint_bounds = ot.Interval(moments, moments)
    bounds = ot.Interval([0.0] * S, [1.0] * S)
    cost = list(sensitivity_values)
    problem = otexp.LinearProblem(cost, bounds, A, constraint_bounds)
    highs = otexp.HiGHS(problem)
    highs.run()
    result = highs.getResult()
    alpha = result.getOptimalPoint()
    return alpha
