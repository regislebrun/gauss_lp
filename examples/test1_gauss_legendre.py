"""Test 1: Recover tensorized Gauss-Legendre quadrature."""

import openturns as ot
import openturns.viewer as otv

from gauss_lp_quadrature import gauss_lp_quadrature

basis = ot.OrthogonalProductPolynomialFactory(
    [ot.LegendreFactory()] * 2, ot.NormInfEnumerateFunction(2)
)
K = 2
polynoms = [basis.build(i) for i in range((2 * K) ** 2)]
N = len(polynoms)
print(f"N = {N}")
mu = basis.getMeasure()
xi, w, best_residual, m, quad_nodes_coll, quad_weights_coll  = gauss_lp_quadrature(mu, polynoms, alphaS=100, Ngauss=100,
                                epsilon=1e-5, verbose=True, show_pdf=False)
print("Nodes:\n", xi)
print("Weights:", w)
print("Moments:", m)
ref = ot.GaussProductExperiment(mu, [K] * 2)
xi_ref, w_ref = ref.generateWithWeights()
print("Reference nodes:\n", xi_ref)
print("Reference weights:", w_ref)
print('')
print('Quadrature collections : ')
print('Nodes - Weights:\n')
for i in range(len(quad_nodes_coll)):
    print('nb nodes=', i+1, "\n", quad_nodes_coll[i])
    print('weights : ', quad_weights_coll[i])

disc_dist = ot.FiniteDiscreteDistribution(xi, w)
graph = disc_dist.drawPDF(1.1*xi.getMin(),1.1*xi.getMax(),[101]*2)
graph.setTitle(
    rf"$U(-1,1)\otimes U(-1,1)$ quadrature: {len(w)} nodes"
)
graph.setXTitle(r'$x_1$')
graph.setYTitle(r'$x_2$')
view = otv.View(graph)
view.show()
