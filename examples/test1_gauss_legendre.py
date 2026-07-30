"""Test 1: Recover tensorized Gauss-Legendre quadrature."""

import openturns as ot

from gauss_lp_quadrature import gauss_lp_quadrature

basis = ot.OrthogonalProductPolynomialFactory(
    [ot.LegendreFactory()] * 2, ot.NormInfEnumerateFunction(2)
)
K = 2
polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
N = len(polynoms)
print(f"N = {N}")
mu = basis.getMeasure()
xi, w, m = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=100,
                                epsilon=1e-5, verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
print("Moments:", m)
ref = ot.GaussProductExperiment(mu, [K] * 2)
xi_ref, w_ref = ref.generateWithWeights()
print("Reference nodes:\n", xi_ref)
print("Reference weights:", w_ref)
