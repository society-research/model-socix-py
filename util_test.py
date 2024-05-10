import util
import pytest
import numpy as np


@pytest.mark.parametrize(
    "name, pos_source, pos_target, collection_list, exp",
    [
        [
            "a single agent (id=1) collects at 0,0, then at 5,5",
            np.array([[0, 0]]),
            np.array([[5, 5]]),
            np.array([[1, 0, 0], [1, 0, 0], [1, 5, 5]]),
            np.array([[1]]),
        ],
        [
            "two humans collecting each from two resources",
            np.array([[0, 0], [1, 1]]),
            np.array([[5, 5], [6, 6]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 5, 5],
                    [2, 1, 1],
                    [2, 6, 6],
                ]
            ),
            np.array(
                [
                    [0.5, 0],
                    [0, 0.5],
                ]
            ),
        ],
        [
            "single human walking back and forth beween resources",
            np.array([[0, 0]]),
            np.array([[5, 5]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 5, 5],
                    [1, 0, 0],
                    [1, 5, 5],
                ]
            ),
            np.array(
                [
                    [1.0],
                ]
            ),
        ],
    ],
)
def test_collected_resource_list_to_cost_matrix(
    name, pos_source, pos_target, collection_list, exp
):
    got = util.collected_resource_list_to_cost_matrix(
        collection_list,
        pos_source,
        pos_target,
    )
    assert exp.shape == got.shape
    assert (exp == got).all()