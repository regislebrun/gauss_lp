"""Test 3: Correlated Normal distribution."""

import openturns as ot
import openturns.viewer as otv

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
xi, w, best_residual, m, quad_nodes_coll, quad_weights_coll = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=100,
                                epsilon=1e-5, verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
for i, p in enumerate(polynoms):
    val = sum(w[j] * p(xi[j])[0] for j in range(len(xi)))
    print(f"  p_{i}: err={abs(val-m[i]):.2e}")

print('')
print('Quadrature collections : ')
print('Nodes - Weights:\n')
for i in range(len(quad_nodes_coll)):
    print('nb nodes=', i+1, "\n", quad_nodes_coll[i])
    print('weights : ', quad_weights_coll[i])

disc_dist = ot.FiniteDiscreteDistribution(xi, w)
graph = disc_dist.drawPDF(xi.getMin(),xi.getMax(),[101]*2)

g = mu.drawPDF(1.1*xi.getMin(),1.1*xi.getMax(),[101]*2)
contour = g.getDrawable(0).getImplementation()
contour.buildDefaultLevels(50)
contour.setColorMapNorm('rank')
contour.setIsFilled(True)
g.setDrawable(0, contour)

g.add(graph)
g.setTitle(
    fr"$N(0,1,\rho=0.5)$ quadrature: {len(w)} nodes"
)
g.setXTitle(r'$x_1$')
g.setYTitle(r'$x_2$')
view = otv.View(g)
ot.Show(graph)
