import util
import pytest
import numpy as np


@pytest.mark.parametrize(
    "name, pos_source, pos_target, collection_list, use_last_only, exp",
    [
        [
            "a single agent (id=1) collects at 0,0, then at 5,5",
            np.array([[0, 0]]),
            np.array([[5, 5]]),
            np.array([[1, 0, 0], [1, 0, 0], [1, 5, 5]]),
            False,
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
            False,
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
            False,
            np.array(
                [
                    [1.0],
                ]
            ),
        ],
        [
            "human collects same resource at different locations: multiple sources",
            np.array([[0, 0], [1, 1]]),
            np.array([[5, 5]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 1, 1],
                    [1, 5, 5],
                ]
            ),
            False,
            np.array(
                [
                    [0.5],
                    [0.5],
                ]
            ),
        ],
        [
            "human collects same resource at different locations: >2 steps between sources",
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[5, 5]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 1, 1],
                    [1, 2, 2],
                    [1, 5, 5],
                ]
            ),
            False,
            np.array(
                [
                    [1 / 3],
                    [1 / 3],
                    [1 / 3],
                ]
            ),
        ],
        [
            "human collects same resource at different locations: multiple targets",
            np.array([[0, 0]]),
            np.array([[5, 5], [6, 6]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 5, 5],
                    [1, 6, 6],
                ]
            ),
            False,
            np.array(
                [
                    [0.5, 0.5],
                ]
            ),
        ],
        [
            "human collects same resource at different locations: multiple targets, but only one used",
            np.array([[0, 0]]),
            np.array([[5, 5], [6, 6]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 5, 5],
                    [1, 6, 6],
                ]
            ),
            True,
            np.array(
                [
                    [1.0, 0.0],
                ]
            ),
        ],
        [
            "human collects same resource at different locations: grid",
            np.array([[0, 0], [1, 1]]),
            np.array([[5, 5], [6, 6]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 1, 1],
                    [1, 5, 5],
                    [1, 6, 6],
                ]
            ),
            False,
            np.array(
                [
                    [0.25, 0.25],
                    [0.25, 0.25],
                ]
            ),
        ],
        [
            "2 paths after one another",
            np.array([[0, 0], [1, 1]]),
            np.array([[5, 5], [6, 6]]),
            np.array(
                [
                    [1, 0, 0],
                    [1, 5, 5],
                    [1, 1, 1],
                    [1, 6, 6],
                ]
            ),
            False,
            np.array(
                [
                    [0.5, 0],
                    [0, 0.5],
                ]
            ),
        ],
    ],
)
def test_collected_resource_list_to_cost_matrix(
    name, pos_source, pos_target, collection_list, use_last_only, exp
):
    got = util.collected_resource_list_to_cost_matrix(
        collection_list,
        pos_source,
        pos_target,
        use_last_only=use_last_only,
    )
    assert exp.shape == got.shape
    assert (exp == got).all()
