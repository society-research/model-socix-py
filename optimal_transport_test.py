import random

import numpy as np
import ot

import optimal_transport as solver


def test__make_distrib_unique():
    random.seed(2)
    orig = [[1, 1], [1, 1], [1, 20]]
    got = solver._make_distrib_unique(orig)
    # np.unique sorts by the axis, and the new element will be in the middle between 1 and 20
    assert (got[0] == [1, 1]).all()
    assert (got[2] == [1, 20]).all()
    assert len(got) == 3


def test_loss():
    loss = solver.loss(
        np.array(
            [
                [1, 2],
                [3, 4],
            ]
        ),
        np.array(
            [
                [0.5, 0],
                [0, 0.5],
            ]
        ),
    )
    assert loss == 2.5
    loss = solver.loss(
        np.array(
            [
                [1, 2],
                [3, 4],
            ]
        ),
        np.array(
            [
                [1, 0],
                [0, 0],
            ]
        ),
    )
    assert loss == 6


def test_HyperParameter():
    p = solver.HyperParameter(min=1, max=10, steps=1)


def test_Optimizer():
    o = solver.Optimizer(n_humans=solver.HyperParameter(min=10, max=100, steps=5))
