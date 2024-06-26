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


def test_collect_resource_depleted_after_multiple_collections():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 5)
    for human in humans:
        human.setVariableInt("x", 0)
        human.setVariableInt("y", 0)
        human.setVariableArrayInt("resources", (1, 0))
        human.setVariableFloat("actionpotential", C.AP_DEFAULT)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 0)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 5
    for human in humans:
        assert human.getVariableArrayInt("resources") == (2, 0), "collected resource"
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 5
    for human in humans:
        assert human.getVariableArrayInt("resources") == (
            2,
            0,
        ), "no collected: resource depleted"
    for _ in range(C.RESOURCE_RESTORATION_TICKS + 1):
        simulation.step()
    simulation.getPopulationData(humans)
    for human in humans:
        assert human.getVariableArrayInt("resources") == (
            3,
            0,
        ), "resource is regrown, and should get collected again"


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
    resources[0].setVariableInt("y", 6)
    exp_path = [
        [0, 1],
        [0, 2],
        [0, 3],
        # [0, 4] and [1, 3] have equal euclidean distance to [3, 6], but our
        # model is discrete and must make a deterministic choice
        [0, 4],
        [1, 4],
        [1, 4],
    ]
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    for step, exp_loc in enumerate(exp_path):
        print(f"[{step}] loc:{exp_loc}")
        simulation.step()
        simulation.getPopulationData(humans)
        assert [humans[0].getVariableInt("x"), humans[0].getVariableInt("y")] == exp_loc
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
    # assert humans[0].getVariableArrayInt("resources") == (0, 0), "still sleeping"
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
