"""Test 4: Correlated Normal distribution, application."""
import openturns as ot

from gauss_lp_quadrature import gauss_lp_quadrature

C = ot.CovarianceMatrix([[7.81e-4, 2.83e-3], [2.83e-3, 1.05e-2]])
mean = [0.986, 4.06]
mu = ot.Normal(mean, C)
print("R=", mu.getCorrelation())
basis = ot.OrthogonalProductPolynomialFactory(
    [ot.StandardDistributionPolynomialFactory(ot.AdaptiveStieltjesAlgorithm(mu.getMarginal(i))) for i in range(2)],
    ot.NormInfEnumerateFunction(2)
)
K = 2
polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
print([str(p) for p in polynoms])
N = len(polynoms)
print(f"N = {N}")
xi, w, m = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=1000,
                                epsilon=1e-5, verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
for i, p in enumerate(polynoms):
    val = sum(w[j] * p(xi[j])[0] for j in range(len(xi)))
    print(f"  p_{i}: err={abs(val-m[i]):.5e}")
