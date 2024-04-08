import pytest
import sys

from sx import make_simulation
from pyflamegpu import pyflamegpu as FLAME
import pyflamegpu
import pyflamegpu.codegen


@pytest.fixture
def cleanup_pyflamegpu():
    """Must be called only once per programm execution"""
    pyflamegpu.cleanup()


def test_collect_resource_no_resource_available():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 1)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 1
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 0
    assert (
        humans[0].getVariableInt("resources") == 1
    ), "no resource available to collect"


def test_collect_resource_success():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 1)
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


def test_move_towards_resource():
    model, simulation, ctx = make_simulation()
    humans = pyflamegpu.AgentVector(ctx.human, 1)
    humans[0].setVariableInt("x", 0)
    humans[0].setVariableInt("y", 0)
    humans[0].setVariableInt("resources", 1)
    resources = pyflamegpu.AgentVector(ctx.resource, 1)
    resources[0].setVariableInt("x", 0)
    resources[0].setVariableInt("y", 10)
    simulation.setPopulationData(resources)
    simulation.setPopulationData(humans)
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans.size() == 1
    assert humans[0].getVariableInt("x") == 0
    assert humans[0].getVariableInt("y") == 1, "moving towards resource"
    assert humans[0].getVariableInt("resources") == 1, "no additional resource in range"
    simulation.step()
    simulation.getPopulationData(humans)
    assert humans[0].getVariableInt("resources") == 1, "no additional resource in range"
    assert humans[0].getVariableInt("y") == 2, "moving towards resource"
    # step step step...
