import random

import pytest
import numpy as np
import ot

import optimal_transport as solver


@pytest.mark.parametrize(
    "name, xs, xt, config, exp",
    [
        [
            "2 resources, 1 human",
            np.array([[1, 1]]),
            np.array([[9, 9]]),
            {"n_humans": 1, "steps": 12},
            np.array([[1]]),
        ],
        # TODO: ["3-1 resources", np.array([[0, 0], [30, 20], [0, 10]]), np.array([[20, 20]])],
    ],
)
def test_solve_ot_with_abm(name, xs, xt, config, exp):
    M, meta = solver.solve_ot_with_abm(xs, xt, **config)
    assert (M == exp).all()


@pytest.mark.parametrize(
    "name, xs, xt, config",
    [
        [
            "2 resources, 1 human",
            np.array([[1, 1]]),
            np.array([[9, 9]]),
            {"n_humans": 1, "steps": 12},
        ],
    ],
)
def test_solve_ot_with_abm(name, xs, xt, config):
    _, meta = solver.solve_ot_with_abm(xs, xt, **config)
    assert len(meta["collected_resources"]) >= 2
    assert len(meta["alive_humans"]) == config["steps"]


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


def test_Optimizer_single_param():
    o = solver.Optimizer(n_humans=solver.HyperParameter(min=10, max=12, steps=1))
    it = o.all()
    assert next(it).n_humans == 10
    assert next(it).n_humans == 11

    o = solver.Optimizer(n_humans=solver.HyperParameter(min=10, max=21, steps=5))
    it = o.all()
    assert next(it).n_humans == 10
    assert next(it).n_humans == 15
    assert next(it).n_humans == 20


# def test_Optimizer_multiple_params():
#    o = solver.Optimizer(n_humans=solver.HyperParameter(min=10, max=12, steps=1))
#    it = o.all()
#    assert next(it) == 10
#    assert next(it) == 11
