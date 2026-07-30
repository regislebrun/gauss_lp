"""Test 4: Correlated Normal distribution, application."""
import openturns as ot
import openturns.viewer as otv

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
xi, w, best_residual, m, quad_nodes_coll, quad_weights_coll = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=1000,
                                epsilon=1e-5, verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
for i, p in enumerate(polynoms):
    val = sum(w[j] * p(xi[j])[0] for j in range(len(xi)))
    print(f"  p_{i}: err={abs(val-m[i]):.5e}")
print('')
print('Quadrature collections : ')
print('Nodes - Weights:\n')
for i in range(len(quad_nodes_coll)):
    print('nb nodes=', i+1, "\n", quad_nodes_coll[i])
    print('weights : ', quad_weights_coll[i])

disc_dist = ot.FiniteDiscreteDistribution(xi, w)
graph = disc_dist.drawPDF(xi.getMin(),xi.getMax(),[101]*2)

g = mu.drawPDF(0.98*xi.getMin(),1.02*xi.getMax(),[101]*2)
contour = g.getDrawable(0).getImplementation()
contour.buildDefaultLevels(50)
contour.setColorMapNorm('rank')
contour.setIsFilled(True)
g.setDrawable(0, contour)

g.add(graph)
g.setTitle(
    fr"$N(\mu, \sigma,\rho=0.988)$ quadrature: {len(w)} nodes"
)
g.setXTitle('a')
g.setYTitle('b')
view = otv.View(g)
ot.Show(graph)



