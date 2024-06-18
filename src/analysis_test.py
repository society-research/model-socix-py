import yaml
import pickle
import io

import pytest
import numpy as np

from analysis import _dump_yaml, analyze_optimality2


@pytest.mark.parametrize(
    "tex, expected",
    [
        (
            {"key1": "value1", "key2": "value2"},
            {"key1": "value1", "key2": "value2"},
        ),  # Test string values
        ({"key1": 1, "key2": 2}, {"key1": 1, "key2": 2}),  # Test integer values
        (  # Test float values
            {"key1": 1.23456789, "key2": 2.3456789},
            {"key1": 1.23456789, "key2": 2.3456789},
        ),
        (
            {
                "OptimalityDifferentSamples": 20,
                "OptimalityMean": 5.189941873678869,
                "OptimalityStd": 1.2049060998014252,
                "ConvergenceStepsTotal": 6000,
            },
            {
                "OptimalityDifferentSamples": 20,
                "OptimalityMean": 5.18994187,
                "OptimalityStd": 1.20490609,
                "ConvergenceStepsTotal": 6000,
            },
        ),
        (  # Test np.float values
            {"key1": np.float64(1.23456789), "key2": np.float64(2.3456789)},
            {"key1": np.float64(1.23456789), "key2": np.float64(2.3456789)},
        ),
    ],
)
def test_dump_yaml(tex, expected):
    output = io.StringIO()
    _dump_yaml(output, tex)
    yaml_str = output.getvalue()
    loaded_data = yaml.safe_load(yaml_str)
    for key, _ in tex.items():
        if isinstance(tex[key], float):
            assert abs(loaded_data[key] - expected[key]) < 1e-8
        else:
            assert loaded_data[key] == expected[key]


#def test_analyze_optimality2():
#    d, s, t = analyze_optimality2("sqlite:///src/hyperparam-optimization-ipynb.sqlite", 5, N_tests=2, M_top=2)
#    assert t == """
#    """

#def test_dumpdata():
#    with open("output/abm-convergence-parts.pickle", "wb") as fd:
#        pickle.dump({"conv_loss": [1, 2, 3], "conv_loss_diff": np.array([0.1, 1e-3])}, fd)
