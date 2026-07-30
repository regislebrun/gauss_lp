"""Test 2: Dirichlet distribution on a simplex."""

import openturns as ot

from gauss_lp_quadrature import gauss_lp_quadrature

basis = ot.OrthogonalProductPolynomialFactory(
    [ot.LegendreFactory()] * 2, ot.NormInfEnumerateFunction(2)
)
mu = ot.Dirichlet([1, 1, 1])
support = ot.Mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
K = 2
polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
N = len(polynoms)
print(f"N = {N}")
xi, w, m = gauss_lp_quadrature(mu, polynoms, domain=support,
                                alphaS=200, Ngauss=100, epsilon=1e-5,
                                verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
