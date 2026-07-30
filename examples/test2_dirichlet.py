"""Test 2: Dirichlet distribution on a simplex."""

import openturns as ot
import openturns.viewer as otv

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
xi, w, best_residual, m, quad_nodes_coll, quad_weights_coll  = gauss_lp_quadrature(mu, polynoms, domain=support,
                                alphaS=200, Ngauss=100, epsilon=1e-5,
                                verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
print('')
print('Quadrature collections : ')
print('Nodes - Weights:\n')
for i in range(len(quad_nodes_coll)):
    print('nb nodes=', i+1, "\n", quad_nodes_coll[i])
    print('weights : ', quad_weights_coll[i])

disc_dist = ot.FiniteDiscreteDistribution(xi, w)
graph = disc_dist.drawPDF(xi.getMin(),xi.getMax(),[101]*2)

g = mu.drawPDF(0.9*xi.getMin(),1.1*xi.getMax(),[101]*2)
contour = g.getDrawable(0).getImplementation()
contour.buildDefaultLevels(50)
contour.setColorMapNorm('rank')
contour.setIsFilled(True)
g.setDrawable(0, contour)

g.add(graph)
g.setTitle(
    f"Dirichlet(1,1,1) quadrature: {len(w)} nodes"
)
g.setXTitle(r'$x_1$')
g.setYTitle(r'$x_2$')
view = otv.View(g)
ot.Show(graph)

