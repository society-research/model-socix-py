import math
import numpy as np
import random

import pytest
import ot
import pyflamegpu

import util
from sx import make_simulation, C


def isclose(a, b) -> bool:
    """We're using float32 in CUDA so increase equal-tolerance a bit."""
    return math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-7)


def test_collect_resource_no_resource_available():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableArrayInt("resources", (1, 0))
    humans[0].setVariableFloat("actionpotential", C.AP_DEFAULT)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 1
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 0
    assert (
        humans[0].getVariableFloat("actionpotential")
        == C.AP_DEFAULT + C.AP_PER_TICK_RESTING
    ), "nothing todo, so human should rest"
    assert humans[0].getVariableArrayInt("resources") == (
        1,
        0,
    ), "no resource available to collect"


def test_collect_resource_success():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableArrayInt("resources", (1, 0))
    humans[0].setVariableFloat("actionpotential", C.AP_DEFAULT)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 0)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 1
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 0
    assert humans[0].getVariableArrayInt("resources") == (2, 0), "collected resource"
    assert isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - C.AP_COLLECT_RESOURCE,
    )


def test_move_towards_resource_1d():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableArrayInt("resources", (1, 0))
    humans[0].setVariableFloat("actionpotential", C.AP_DEFAULT)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 5)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 1
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 1, "moving towards resource"
    assert humans[0].getVariableArrayInt("resources") == (
        1,
        0,
    ), "no additional resource in range"
    assert isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - 1 * C.AP_MOVE,
    )
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("y") == 2, "moving towards resource"
    assert humans[0].getVariableArrayInt("resources") == (
        1,
        0,
    ), "no additional resource in range"
    assert isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - 2 * C.AP_MOVE,
    )
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("y") == 2, "resource is already in range"
    assert isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - 2 * C.AP_MOVE - C.AP_COLLECT_RESOURCE,
    )
    assert humans[0].getVariableArrayInt("resources") == (2, 0), "collected resource"


def test_move_towards_resource_2d():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableArrayInt("resources", (1, 0))
    humans[0].setVariableFloat("actionpotential", C.AP_DEFAULT)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 3)
    resources[0].setVariableInt("y", 3)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 1
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("x") == 1
    assert humans[0].getVariableInt("y") == 1
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("x") == 1
    assert humans[0].getVariableInt("y") == 1
    assert humans[0].getVariableArrayInt("resources") == (2, 0)


def test_recover_actionpotential_by_sleeping():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableArrayInt("resources", (0, 0))
    humans[0].setVariableFloat("actionpotential", 0)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 0)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableArrayInt("resources") == (
        0,
        0,
    ), "no AP to collect the resource!"
    assert (
        humans[0].getVariableFloat("actionpotential") == C.AP_PER_TICK_RESTING
    ), "resting restored some AP"
    simulation.step()
    simulation.getPopulationData(humans)
    # TODO(maybe): add agent variable sleeping to count down ~8 hours
    # assert humans[0].getVariableArrayInt("resources") == (0, 0), "still sleepnig"
    # assert (
    #     humans[0].getVariableFloat("actionpotential") == 2 * C.AP_PER_TICK_RESTING
    # ), "resting restored some AP"
    assert humans[0].getVariableArrayInt("resources") == (1, 0), "got enough AP now"


def test_crowding_reduces_actionpotential():
    model, simulation, ctx = make_simulation()
    # required to seed, since this test uses random-walk
    simulation.SimulationConfig().random_seed = 0
    humans = pyflamegpu.AgentVector(ctx.human, 12)
    for human in humans:
        human.setVariableInt("x", 0)
        human.setVariableInt("y", 0)
        human.setVariableArrayInt("resources", (0, 0))
        human.setVariableFloat("actionpotential", C.AP_DEFAULT)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    for human in humans:
        assert human.getVariableInt("is_crowded") == 1
        assert isclose(
            human.getVariableFloat("actionpotential"),
            C.AP_DEFAULT - C.AP_REDUCTION_BY_CROWDING - C.AP_MOVE,
        ), "AP reduced & move happened"
        assert (
            human.getVariableInt("x") != 0 or human.getVariableInt("y") != 0
        ), "crowded humans should random walk"
    simulation.step()
    simulation.getPopulationData(humans)
    for human in humans:
        assert human.getVariableInt("is_crowded") == 0


def test_crowding_should_resolve():
    model, simulation, ctx = make_simulation(grid_size=100)
    # required to seed, since this test uses random-walk
    simulation.SimulationConfig().random_seed = 2
    humans = pyflamegpu.AgentVector(ctx.human, 100)
    for human in humans:
        human.setVariableInt("x", 5)
        human.setVariableInt("y", 5)
    simulation.setPopulationData(humans)
    for _ in range(10):
        simulation.step()
    simulation.getPopulationData(humans)
    crowded = 0
    for human in humans:
        crowded += human.getVariableInt("is_crowded")
    assert crowded <= 15, "<= 15% should be crowded after 10 steps"


def test_starve_without_resources():
    model, simulation, ctx = make_simulation(grid_size=100)
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableArrayInt("resources", (0, 0))
    humans[0].setVariableFloat("actionpotential", 0)
    simulation.setPopulationData(humans)
    for _ in range(C.HUNGER_STARVED_TO_DEATH + 1):
        simulation.step()
    simulation.getPopulationData(humans)
    assert len(humans) == 0, "agent is dead, no more agents left"


def test_require_2_different_resources_for_survival():
    model, simulation, ctx = make_simulation(grid_size=100)
    humans = pyflamegpu.AgentVector(ctx.human, 2)
    humans[0].setVariableInt("x", 1)
    humans[0].setVariableInt("y", 1)
    humans[0].setVariableArrayInt("resources", (1, 0))
    humans[0].setVariableFloat("actionpotential", C.AP_DEFAULT)
    humans[1].setVariableInt("x", 1)
    humans[1].setVariableInt("y", 1)
    humans[1].setVariableArrayInt("resources", (1, 1))
    humans[1].setVariableFloat("actionpotential", C.AP_DEFAULT)
    simulation.setPopulationData(humans)
    for _ in range(C.HUNGER_STARVED_TO_DEATH + 1):
        simulation.step()
    simulation.getPopulationData(humans)
    assert len(humans) == 1, "one starves, one stays alive"


def test_move_towards_2nd_resource_to_stay_alive():
    model, simulation, ctx = make_simulation(grid_size=100)
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableArrayInt("resources", (10, 0))
    humans[0].setVariableFloat("actionpotential", C.AP_DEFAULT)
    resources = pyflamegpu.AgentVector(ctx.resource, 2)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 0)
    resources[0].setVariableInt("type", 0)
    resources[1].setVariableInt("x", 0)
    resources[1].setVariableInt("y", 5)
    resources[1].setVariableInt("type", 1)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 1
    assert isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - C.AP_MOVE,
    )
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 2
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableArrayInt("resources") == (10, 1)


@pytest.mark.parametrize(
    "name, pos_source, pos_target",
    [
        ["single resource, on y axis", np.array([[11, 11]]), np.array([[0, 30]])],
        ["3-1 resources", np.array([[0, 0], [30, 20], [0, 10]]), np.array([[20, 20]])],
    ],
)
def test_solve_ot_problem(name, pos_source, pos_target):
    """Setup here is similar to https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html"""
    # this test initializes humans at "random" positions
    # XXX: should be an accuracy benchmark to repeat this with different seeds!
    seed = 2
    random.seed(seed)
    # OT solution: EMD
    M_loss = ot.dist(pos_source, pos_target)
    n, m = len(pos_source), len(pos_target)
    a, b = np.ones((n,)) / n, np.ones((m,)) / m
    exp = ot.emd(a, b, M_loss)
    # ABM solution:
    grid_size = 30
    model, simulation, ctx = make_simulation(grid_size=grid_size)
    simulation.SimulationConfig().random_seed = seed
    resources = pyflamegpu.AgentVector(ctx.resource, len(pos_source) + len(pos_target))
    for i, p in enumerate(pos_source):
        resources[i].setVariableInt("x", int(p[0]))
        resources[i].setVariableInt("y", int(p[1]))
        resources[i].setVariableInt("type", 0)
    for i, p in enumerate(pos_target):
        resources[i + len(pos_source)].setVariableInt("x", int(p[0]))
        resources[i + len(pos_source)].setVariableInt("y", int(p[1]))
        resources[i + len(pos_source)].setVariableInt("type", 1)
    n_humans = 10
    humans = pyflamegpu.AgentVector(ctx.human, n_humans)
    for human in humans:
        human.setVariableInt("x", random.randint(0, grid_size))
        human.setVariableInt("y", random.randint(0, grid_size))
        human.setVariableArrayInt("resources", (2, 2))
        human.setVariableFloat("actionpotential", C.AP_DEFAULT)
    for av in [resources, humans]:
        simulation.setPopulationData(av)
    steps = 100
    paths = []
    collected_resources = []
    for step in range(steps):
        simulation.step()
        simulation.getPopulationData(humans)
        for human in humans:
            id = human.getID()
            x, y = human.getVariableInt("x"), human.getVariableInt("y")
            paths.append([id, x, y])
            loc = human.getVariableArrayInt("ana_last_resource_location")
            if loc != (-1, -1):
                collected_resources.append([id, *loc])
    got = util.collected_resource_list_to_cost_matrix(
        collected_resources, pos_source, pos_target
    )
    assert exp.shape == got.shape
    print("res[-10:]:\n", collected_resources[-10:])
    print("M_loss:\n", M_loss)
    print("exp (EMD):\n", exp)
    print("got (ABM):\n", got)
    assert (
        (got != 0) == (exp != 0)
    ).all(), "assume the non-zero-entries are the same positions"
    # TODO: enable allclose comparison again
    # assert np.allclose(exp, got, atol=0.1)
