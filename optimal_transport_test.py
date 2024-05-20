import random

import optimal_transport as solver


def test__make_distrib_unique():
    random.seed(2)
    orig = [[1, 1], [1, 1], [1, 2]]
    got = solver._make_distrib_unique(orig, 1, 10)
    assert (got[:2] == [[1, 1], [1, 2]]).all()
    assert len(got) == 2  # TODO should be 3


def test_HyperParameter():
    p = solver.HyperParameter(min=1, max=10, steps=1)


def test_Optimizer():
    o = solver.Optimizer(n_humans=solver.HyperParameter(min=10, max=100, steps=5))
