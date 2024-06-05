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


@pytest.mark.parametrize(
    "name, M",
    [
        [
            "np.random.rand(4, 4)",
            np.array(
                [
                    [0.47336145, 0.29978534, 0.59259927, 0.16709003],
                    [0.22697437, 0.95703714, 0.22964212, 0.27628888],
                    [0.30351254, 0.75309368, 0.85352262, 0.11278805],
                    [0.5728611, 0.29681332, 0.74419422, 0.47940583],
                ]
            ),
        ],
        [
            "np.random.rand(4, 4) with 1 zero",
            np.array(
                [
                    [0.47336145, 0.29978534, 0.59259927, 0.16709003],
                    [0.22697437, 0, 0.22964212, 0.27628888],
                    [0.30351254, 0.75309368, 0.85352262, 0.11278805],
                    [0.5728611, 0.29681332, 0.74419422, 0.47940583],
                ]
            ),
        ],
        [
            "np.random.rand(4, 4) with many zeros",
            np.array(
                [
                    [0.47336145, 0.29978534, 0.59259927, 0.16709003],
                    [0.22697437, 0, 0.22964212, 0],
                    [0.30351254, 0.75309368, 0.85352262, 0],
                    [0.5728611, 0.29681332, 0.74419422, 0],
                ]
            ),
        ],
        [
            "np.random.rand(4, 4), but whole row zero",
            np.array(
                [
                    [0, 0, 0, 0],
                    [0.22697437, 0, 0.22964212, 0.123],
                    [0.30351254, 0.75309368, 0.85352262, 0.001],
                    [0.5728611, 0.29681332, 0.74419422, 0.5],
                ]
            ),
        ],
        [
            "np.random.rand(4, 4), but whole column zero",
            np.array(
                [
                    [0.1, 0.1, 0.9, 0],
                    [0.22697437, 0, 0.22964212, 0],
                    [0.30351254, 0.75309368, 0.85352262, 0],
                    [0.5728611, 0.29681332, 0.74419422, 0],
                ]
            ),
        ],
        [
            "np.random.rand(1, 2)",
            np.array([[0.71554484, 0.47968838]]),
        ],
    ],
)
def test_doubly_stochastic(name, M):
    M /= M.sum()
    print("Initial matrix:")
    print(M)
    M = util.doubly_stochastic(M)
    print("Normalized matrix:")
    print(M)
    print("Row sums:", M.sum(axis=1))
    print("Column sums:", M.sum(axis=0))
    assert np.allclose(
        M.sum(axis=1), [1 / len(M)] * len(M[0]), atol=0.01
    ), "row sums must be even"
    assert np.allclose(
        M.sum(axis=0), [1 / len(M[0])] * len(M), atol=0.01
    ), "column sums must be even"


def test__fix_zeros():
    M = np.array(
        [
            [0, 0, 0, 0],
            [0.22697437, 0, 0.22964212, 0.123],
            [0.30351254, 0.75309368, 0.85352262, 0.001],
            [0.5728611, 0.29681332, 0.74419422, 0.5],
        ]
    )
    assert np.allclose(
        util._fix_zeros(M),
        np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.22697437, 0, 0.22964212, 0.123],
                [0.30351254, 0.75309368, 0.85352262, 0.001],
                [0.5728611, 0.29681332, 0.74419422, 0.5],
            ]
        ),
    )
    M = np.array(
        [
            [0, 0.1, 0, 0],
            [0.22697437, 0, 0.22964212, 0],
            [0.30351254, 0.75309368, 0.85352262, 0],
            [0.5728611, 0.29681332, 0.74419422, 0],
        ]
    )
    assert np.allclose(
        util._fix_zeros(M),
        np.array(
            [
                [0, 0.1, 0, 0.25],
                [0.22697437, 0, 0.22964212, 0.25],
                [0.30351254, 0.75309368, 0.85352262, 0.25],
                [0.5728611, 0.29681332, 0.74419422, 0.25],
            ]
        ),
    )
