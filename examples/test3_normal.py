"""Test 3: Correlated Normal distribution."""

import openturns as ot

from gauss_lp_quadrature import gauss_lp_quadrature

basis = ot.OrthogonalProductPolynomialFactory(
    [ot.HermiteFactory()] * 2, ot.NormInfEnumerateFunction(2)
)
rho = 0.5
R = ot.CorrelationMatrix(2, [1.0, rho, rho, 1.0])
mu = ot.Normal([0.0] * 2, [1.0] * 2, R)
K = 2
polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
N = len(polynoms)
print(f"N = {N}")
xi, w, m = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=100,
                                epsilon=1e-5, verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
for i, p in enumerate(polynoms):
    val = sum(w[j] * p(xi[j])[0] for j in range(len(xi)))
    print(f"  p_{i}: err={abs(val-m[i]):.2e}")
