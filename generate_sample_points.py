import openturns as ot


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
                a = domain_out.getLowerBound()
                b = domain_out.getUpperBound()
                sampling_dist = ot.JointDistribution(
                    [ot.Uniform(a[i], b[i]) for i in range(dimension)]
                )
    XS = sampling_dist.getSample(S)
    return XS, a, b, domain_out
