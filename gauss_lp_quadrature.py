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

import openturns as ot
import openturns.experimental as otexp
import numpy as np

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

def compute_moments(measure, polynoms, domain=None, Ngauss=100):
    """Compute the moments int_Omega p_k(x) pdf(x) dx for each polynomial."""
    pdf = measure.getPDF()
    dimension = measure.getDimension()
    moments = ot.Point(0)
    if domain is None:
        box = measure.getRange()
        integration = ot.GaussLegendre([Ngauss] * dimension)
        for p in polynoms:
            moments.add(integration.integrate(pdf * p, box)[0])
    else:
        integration = ot.SimplicialCubature()
        integration.setMaximumCallsNumber(Ngauss ** dimension)
        for p in polynoms:
            moments.add(integration.integrate(pdf * p, domain)[0])
    return moments


def generate_sample_points(measure, domain, S, dimension):
    """Generate S sample points distributed uniformly over the domain."""
    if domain is None:
        D = measure.getRange()
        a = D.getLowerBound()
        b = D.getUpperBound()
        sampling_dist = ot.JointDistribution(
            [ot.Uniform(a[i], b[i]) for i in range(dimension)]
        )
        domain_out = D
    else:
        if isinstance(domain, ot.Mesh):
            domain_out = ot.MeshDomain(domain)
            sampling_dist = ot.UniformOverMesh(domain)
            mesh = domain
            a = mesh.getVertices().getMin()
            b = mesh.getVertices().getMax()
        else:
            domain_out = domain
            if isinstance(domain_out, ot.MeshDomain):
                mesh = domain_out.getMesh()
                a = mesh.getVertices().getMin()
                b = mesh.getVertices().getMax()
                sampling_dist = ot.UniformOverMesh(mesh)
            else:
                # fallback: uniform in bounding box
                a = domain_out.getLowerBound()
                b = domain_out.getUpperBound()
                sampling_dist = ot.JointDistribution(
                    [ot.Uniform(a[i], b[i]) for i in range(dimension)]
                )
    XS = sampling_dist.getSample(S)
    return XS, a, b, domain_out


def solve_lp(XS, polynoms, moments, sensitivity_values):
    """Solve the discretized LP (eq. 3 in Ryu 2013).

    Returns the vector of weights (alpha_i >= 0) for each sample point.
    """
    S = len(XS)
    n = len(polynoms)
    # Constraint matrix A (n x M): A[i,j] = p_i(s_j)
    A = ot.DesignProxy(XS, polynoms).computeDesign(list(range(n))).transpose()
    # Equality constraints: A * alpha == moments
    constraint_bounds = ot.Interval(moments, moments)
    # Variable bounds: 0 <= alpha_i <= 1
    bounds = ot.Interval([0.0] * S, [1.0] * S)
    # Cost: sensitivity function evaluated at each sample point
    cost = list(sensitivity_values)
    problem = otexp.LinearProblem(cost, bounds, A, constraint_bounds)
    highs = otexp.HiGHS(problem)
    highs.run()
    result = highs.getResult()
    alpha = result.getOptimalPoint()
    return alpha


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

    # Try from N/(d+1) clusters up to n_support
    # Try from 1 cluster up to n_support
    n_min = 1#max(N // (dimension + 1), 1)
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

        # Build objective: sum of squared moment residuals
        # Variables: [w_0..w_{size-1}, x_0..x_{size*dimension-1}]
        # For non-box domains, penalize points outside the domain
        need_domain_check = (domain is not None)

        if with_mean:
            # Find which of the starting points is the closest to the mean
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
                # Penalize the distance of the first node to the mean of the measure
                value += (meanMeasure - XS_eval[iMean]).normSquare()
            return [value]

        objective = ot.PythonFunction(size + flat_size, 1, objective_py)

        problem = ot.OptimizationProblem(objective)
        # Bounds: weights in [0, 1], nodes in domain box
        lb_weights = [0.0] * size
        ub_weights = [1.0] * size
        lb_nodes = list(a_lower) * size
        ub_nodes = list(b_upper) * size
        bounds = ot.Interval(lb_weights + lb_nodes, ub_weights + ub_nodes)
        problem.setBounds(bounds)

        algo = ot.TNC(problem)
        # Cobyla needs enough evaluations: ~1000 per variable for convergence
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


if __name__ == "__main__":
    # Test 1: Recover tensorized Gauss-Legendre quadrature
    print("#" * 50)
    print("Test 1: Recover tensorized Gauss-Legendre quadrature")
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()] * 2, ot.NormInfEnumerateFunction(2)
    )
    K = 3
    polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
    N = len(polynoms)
    print(f"N = {N}")
    mu = basis.getMeasure()
    xi, w, m = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=100,
                                   epsilon=1e-5, verbose=True, show_pdf=True)
    print("Nodes:\n", xi)
    print("Weights:", w)
    print("Moments:", m)
    ref = ot.GaussProductExperiment(mu, [K] * 2)
    xi_ref, w_ref = ref.generateWithWeights()
    print("Reference nodes:\n", xi_ref)
    print("Reference weights:", w_ref)

    # Test 2: Dirichlet distribution on a simplex
    print("\n" + "#" * 50)
    print("Test 2: Dirichlet distribution on a simplex")
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()] * 2, ot.NormInfEnumerateFunction(2)
    )
    mu = ot.Dirichlet([1, 1, 1])
    support = ot.Mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    K = 3
    polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
    N = len(polynoms)
    print(f"N = {N}")
    xi, w, m = gauss_lp_quadrature(mu, polynoms, domain=support,
                                   alphaS=200, Ngauss=100, epsilon=1e-5,
                                   verbose=True, show_pdf=True)
    print("Nodes:\n", xi)
    print("Weights:", w)

    # Test 3: Correlated Normal distribution
    print("\n" + "#" * 50)
    print("Test 3: Correlated Normal distribution")
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.HermiteFactory()] * 2, ot.NormInfEnumerateFunction(2)
    )
    rho = 0.5
    R = ot.CorrelationMatrix(2, [1.0, rho, rho, 1.0])
    mu = ot.Normal([0.0] * 2, [1.0] * 2, R)
    K = 3
    polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
    N = len(polynoms)
    print(f"N = {N}")
    xi, w, m = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=100,
                                   epsilon=1e-5, verbose=True, show_pdf=True)
    print("Nodes:\n", xi)
    print("Weights:", w)
    for i, p in enumerate(polynoms):
        val = sum(w[j] * p(xi[j])[0] for j in range(len(xi)))
        print(f"  p_{i}: err={abs(val-m[i]):.2e}")

    # Test 4: Correlated Normal distribution, application
    print("\n" + "#" * 50)
    print("Test 3: Correlated Normal distribution")
    C = ot.CovarianceMatrix([[7.81e-4, 2.83e-3], [2.83e-3, 1.05e-2]])
    mean = [0.986, 4.06]
    mu = ot.Normal(mean, C)
    print("R=", mu.getCorrelation())
    basis = ot.OrthogonalProductPolynomialFactory([ot.StandardDistributionPolynomialFactory(ot.AdaptiveStieltjesAlgorithm(mu.getMarginal(i))) for i in range(2)], ot.NormInfEnumerateFunction(2))
    K = 3
    polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
    print([str(p) for p in polynoms])
    N = len(polynoms)
    print(f"N = {N}")
    xi, w, m = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=1000,
                                   epsilon=1e-5, verbose=True, show_pdf=True)
    print("Nodes:\n", xi)
    print("Weights:", w)
    for i, p in enumerate(polynoms):
        val = sum(w[j] * p(xi[j])[0] for j in range(len(xi)))
        print(f"  p_{i}: err={abs(val-m[i]):.5e}")
