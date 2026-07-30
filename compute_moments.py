import openturns as ot


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
