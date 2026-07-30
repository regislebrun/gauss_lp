"""Gauss-LP quadrature following Ryu & Boyd (2013), Section 4.

Algorithm:
1. Compute moments of reference measure against test functions
2. Discretize domain with a fine grid of random sample points
3. Solve LP: min sum_i w_i r(x_i)  s.t. sum_i w_i p_k(x_i) = mu(p_k), w_i >= 0
4. Cluster the support of the LP solution
5. Refine nodes/weights via nonlinear least-squares with Cobyla

References:
  Ryu, E.K. and Boyd, S.P., "Extensions of Gauss Quadrature Via Linear
  Programming", Found Comput Math (2015) 15: 953-971.
"""

import numpy as np

import openturns as ot
import openturns.experimental as otexp

from compute_moments import compute_moments
from generate_sample_points import generate_sample_points
from solve_lp import solve_lp
from cluster_rule import cluster_rule
from build_cluster_path import build_cluster_path

ot.ResourceMap.SetAsScalar("OptimizationAlgorithm-DefaultMaximumAbsoluteError", 1.0e-12)
ot.ResourceMap.SetAsScalar("OptimizationAlgorithm-DefaultMaximumConstraintError", 1.0e-12)
ot.ResourceMap.SetAsScalar("OptimizationAlgorithm-DefaultMaximumRelativeError", 1.0e-12)
ot.ResourceMap.SetAsScalar("OptimizationAlgorithm-DefaultMaximumResidualError", 1.0e-12)
ot.ResourceMap.SetAsScalar("OptimizationAlgorithm-DefaultMaximumTimeDuration", -1.0)
ot.ResourceMap.SetAsUnsignedInteger("OptimizationAlgorithm-DefaultMaximumCallsNumber", 10000)
ot.ResourceMap.SetAsUnsignedInteger("OptimizationAlgorithm-DefaultMaximumIterationNumber", 10000)
ot.ResourceMap.AddAsBool('HiGHS-output_flag', False)
ot.ResourceMap.AddAsUnsignedInteger('HiGHS-threads', 32)
ot.ResourceMap.SetAsScalar("Distribution-DiscreteDrawPDFScaling", 0.025)
ot.ResourceMap.SetAsBool("Distribution-ShowSupportDiscretePDF", True)
ot.ResourceMap.SetAsString("Distribution-SupportPointStyleDiscretePDF", "bullet")
ot.ResourceMap.SetAsBool("Distribution-ScaleColorsDiscretePDF", True)


def gauss_lp_quadrature(measure, polynoms, sensitivity=None, domain=None,
                        alphaS=100, Ngauss=100, epsilon=5e-5, with_mean=True,
                        verbose=False, show_pdf=False):
    """Compute a Gauss-LP quadrature following Ryu & Boyd (2013).

    Parameters
    ----------
    measure : ot.Distribution
        Reference measure with respect to which moments are computed.
    polynoms : list of ot.Function
        Test functions p_0, ..., p_{n-1} that the quadrature must integrate
        exactly (p_0 should be the constant function 1).
    sensitivity : ot.Function, optional
        Sensitivity function r(x). Minimized by the LP to induce sparsity.
        Default: r(x) = 1 (minimizes total weight sum).
    domain : ot.Interval or ot.Mesh, optional
        Integration domain. If None, uses measure.getRange().
    alphaS : int
        Oversampling factor: number of LP variables = alphaS * len(polynoms).
    Ngauss : int
        Number of Gauss-Legendre points per dimension for moment computation.
    epsilon : float
        Convergence tolerance for the refinement step (squared residual).
    with_mean : bool
        If True, force the mean of the measure to be one of the integration points.
    verbose : bool
        If True, print progress information.
    show_pdf : bool
        If True, display the resulting weighted sample as a
        FiniteDiscreteDistribution PDF plot.

    Returns
    -------
    XS : ot.Sample
        Quadrature nodes (N x dimension).
    w : ot.Point
        Quadrature weights.
    moments : ot.Point
        Reference moments.
    """
    dimension = measure.getDimension()
    N = len(polynoms)
    S = N * alphaS
    if with_mean:
        meanMeasure = measure.getMean()

    # ---- Step 1: compute moments ----
    moments = compute_moments(measure, polynoms, domain, Ngauss)
    if verbose:
        print("Moments =", moments)

    # ---- Step 2: generate sample points ----
    XS, a_lower, b_upper, domain_obj = generate_sample_points(
        measure, domain, S, dimension
    )

    # ---- Step 3: evaluate sensitivity and solve LP ----
    if sensitivity is None:
        r_vals = ot.Point(S, 1.0)
        if with_mean:
            iMean = 0
            dMean = (meanMeasure - XS[0]).normSquare()
            for i in range(1, len(XS)):
                d = (meanMeasure - XS[i]).normSquare()
                if d < dMean:
                    dMean = d
                    iMean = i
            r_vals[iMean] = 0
    else:
        r_vals = ot.Point(S)
        for i in range(S):
            r_vals[i] = sensitivity(XS[i])[0]

    alpha = solve_lp(XS, polynoms, moments, r_vals)
    w = ot.Point([alpha[i] for i in range(S) if alpha[i] > 0.0])
    XS_support = ot.Sample([XS[i] for i in range(S) if alpha[i] > 0.0])
    n_support = len(w)
    if verbose:
        print(f"LP solution has {n_support} support points (out of {S})")

    # ---- Step 4+5: cluster and refine ----
    path = build_cluster_path(XS_support, w)

    n_min = 1
    best_residual = ot.SpecFunc.MaxScalar
    best_nodes = None
    best_weights = None
    converged = False

    for n_clusters in range(n_min, n_support + 1):
        if verbose:
            print(f"--- Trying {n_clusters} clusters ---")

        if n_clusters < n_support:
            XSi, wi = cluster_rule(XS_support, w, path, n_clusters)
        else:
            XSi, wi = XS_support, w

        size = len(XSi)
        flat_size = size * dimension

        need_domain_check = (domain is not None)

        if with_mean:
            iMean = 0
            dMean = (meanMeasure - XSi[0]).normSquare()
            for ic in range(1, len(XSi)):
                d = (meanMeasure - XSi[ic]).normSquare()
                if d < dMean:
                    d = dMean
                    iMean = ic

        def objective_py(X):
            Xpt = ot.Point(X)
            w_part = Xpt[:size]
            data = Xpt[size:]
            XS_eval = ot.Sample(
                np.array(data).reshape((size, dimension))
            )
            if need_domain_check:
                inside = domain_obj.contains(XS_eval)
                if sum(inside) < size:
                    return [ot.SpecFunc.MaxScalar]
            phi = ot.DesignProxy(XS_eval, polynoms).computeDesign(
                list(range(len(polynoms)))
            ).transpose()
            residual = phi * w_part - moments
            value = residual.normSquare()
            if with_mean:
                value += (meanMeasure - XS_eval[iMean]).normSquare()
            return [value]

        objective = ot.PythonFunction(size + flat_size, 1, objective_py)

        problem = ot.OptimizationProblem(objective)
        lb_weights = [0.0] * size
        ub_weights = [1.0] * size
        lb_nodes = list(a_lower) * size
        ub_nodes = list(b_upper) * size
        bounds = ot.Interval(lb_weights + lb_nodes, ub_weights + ub_nodes)
        problem.setBounds(bounds)

        algo = ot.TNC(problem)
        max_calls = max(10000, 1000 * (size + flat_size))
        algo.setMaximumCallsNumber(max_calls)
        algo.setMaximumIterationNumber(int(0.5 * max_calls))

        x0 = list(wi)
        for x in XSi:
            x0 += list(x)
        algo.setStartingPoint(x0)
        try:
            algo.run()
            result = algo.getResult()
            x_final = result.getOptimalPoint()
            residual_sq = objective(x_final)[0]
        except Exception as e:
            if verbose:
                print(f"  {algo.getClassName()} failed: {e}")
            continue

        if verbose:
            print(f"  Residual^2 = {residual_sq:.2e}")

        if residual_sq < best_residual:
            best_residual = residual_sq
            best_weights = ot.Point(x_final[:size])
            best_nodes = ot.Sample(
                np.array(x_final[size:]).reshape((size, dimension))
            )

        if residual_sq < epsilon * epsilon:
            converged = True
            if verbose:
                print(f"  Converged with {n_clusters} nodes")
            break

    if verbose:
        if converged:
            print(f"Gauss-LP quadrature found with {len(best_weights)} nodes")
        else:
            print(f"No convergence, best residual^2 = {best_residual:.2e} "
                  f"with {len(best_weights)} nodes")

    if show_pdf:
        total_w = sum(best_weights)
        probabilities = [w / total_w for w in best_weights]
        disc_dist = ot.FiniteDiscreteDistribution(best_nodes, probabilities)
        graph = disc_dist.drawPDF(measure.getRange().getLowerBound(), measure.getRange().getUpperBound())
        if with_mean:
            pt = ot.Cloud([meanMeasure])
            pt.setColor("white")
            pt.setPointStyle("fsquare")
            pt.setLegend("")
            graph.add(pt)
            pt.setColor("black")
            pt.setPointStyle("bullet")
            pt.setLegend("Mean point")
            graph.add(pt)
        graph.setTitle(
            f"FiniteDiscreteDistribution PDF ({len(best_weights)} nodes, "
            f"{dimension}D)"
        )
        ot.Show(graph)

    return best_nodes, best_weights, moments


