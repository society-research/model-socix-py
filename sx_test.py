import math
import pyflamegpu

from sx import make_simulation, C


def test_collect_resource_no_resource_available():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 1)
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
    assert (
        humans[0].getVariableInt("resources") == 1
    ), "no resource available to collect"


def test_collect_resource_success():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 1)
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
    assert humans[0].getVariableInt("resources") == 2, "collected resource"
    assert math.isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - C.AP_COLLECT_RESOURCE,
        rel_tol=1e-7,
    )


def test_move_towards_resource_1d():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 1)
    humans[0].setVariableFloat("actionpotential", C.AP_DEFAULT)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 4)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 1
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 1, "moving towards resource"
    assert humans[0].getVariableInt("resources") == 1, "no additional resource in range"
    assert math.isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - 1 * C.AP_MOVE,
        rel_tol=1e-7,
    )
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("y") == 2, "moving towards resource"
    assert humans[0].getVariableInt("resources") == 1, "no additional resource in range"
    assert math.isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - 2 * C.AP_MOVE,
        rel_tol=1e-7,
    )
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("y") == 2, "resource is already in range"
    assert math.isclose(
        humans[0].getVariableFloat("actionpotential"),
        C.AP_DEFAULT - 2 * C.AP_MOVE - C.AP_COLLECT_RESOURCE,
        rel_tol=1e-7,
    )
    assert humans[0].getVariableInt("resources") == 2, "collected resource"


def test_move_towards_resource_2d():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 1)
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
    assert humans[0].getVariableInt("resources") == 2


def test_recover_actionpotential_by_sleeping():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 0)
    humans[0].setVariableFloat("actionpotential", 0)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 0)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("resources") == 0, "no AP to collect the resource!"
    assert (
        humans[0].getVariableFloat("actionpotential") == C.AP_PER_TICK_RESTING
    ), "resting restored some AP"
    simulation.step()
    simulation.getPopulationData(humans)
    # TODO(maybe): add agent variable sleeping to count down ~8 hours
    # assert humans[0].getVariableInt("resources") == 0, "still sleepnig"
    # assert (
    #     humans[0].getVariableFloat("actionpotential") == 2 * C.AP_PER_TICK_RESTING
    # ), "resting restored some AP"
    assert humans[0].getVariableInt("resources") == 1, "got enough AP now"


def test_crowding_reduces_actionpotential():
    model, simulation, ctx = make_simulation()
    # required to seed, since this test uses random-walk
    simulation.SimulationConfig().random_seed = 0
    humans = pyflamegpu.AgentVector(ctx.human, 12)
    for human in humans:
        human.setVariableInt("x", 0)
        human.setVariableInt("y", 0)
        human.setVariableInt("resources", 0)
        human.setVariableFloat("actionpotential", C.AP_DEFAULT)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    for human in humans:
        assert human.getVariableInt("is_crowded") == 1
        assert math.isclose(
            human.getVariableFloat("actionpotential"),
            C.AP_DEFAULT - C.AP_REDUCTION_BY_CROWDING - C.AP_MOVE,
            abs_tol=1e-7,
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
        # print(
        #     f"human[{human.getID()}], at=[{human.getVariableInt('x')}, {human.getVariableInt('y')}], ap={human.getVariableFloat('actionpotential')}"
        # )
        crowded += human.getVariableInt("is_crowded")
    assert crowded <= 15, "<= 15% should be crowded after 10 steps"


# def test_movement_around_2d_grid_boundaries():
#    model, simulation, ctx = make_simulation()
#    humans = pyflamegpu.AgentVector(ctx.human, 1)
#    humans[0].setVariableInt("x", 0)
#    humans[0].setVariableInt("y", 0)
#    simulation.setPopulationData(humans)
#    simulation.step()
#    simulation.getPopulationData(humans)
