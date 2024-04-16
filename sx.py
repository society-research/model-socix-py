import sys
import random
import math
import ostruct
import pyflamegpu
import pyflamegpu.codegen


def sqbrt(x):
    root = abs(x) ** (1 / 2)
    return root if x >= 0 else -root


C = ostruct.OpenStruct(
    AGENT_COUNT=64,
    RESOURCE_COLLECTION_RANGE=3.0,
    # TODO: can be removed, right?!
    HUMAN_MOVE_RANGE=10.0,
    # default amount of action potential (AP)
    AP_DEFAULT=1.0,
    # AP needed to collect a resource
    AP_COLLECT_RESOURCE=0.1,
    # AP needed to move a block
    AP_MOVE=0.05,
    # required sleep per night in hours
    SLEEP_REQUIRED_PER_NIGHT=8,
    # amount of humans in a single tile, after which humans feel crowded
    # (reduced AP)
    N_HUMANS_CROWDED=10,
)
# 2D space bondaries
C.ENV_MAX = math.floor(sqbrt(C.AGENT_COUNT))
# Restored AP per tick spend resting.
C.AP_PER_TICK_RESTING = C.AP_DEFAULT / C.SLEEP_REQUIRED_PER_NIGHT
# AP reduction caused by crowding, see @N_HUMANS_CROWDED
# Note: humans should be able to move, even with reduced AP by crowding
C.AP_REDUCTION_BY_CROWDING = C.AP_DEFAULT - C.AP_MOVE


@pyflamegpu.device_function
def vec2Length(x: int, y: int) -> float:
    return math.sqrtf(x * x + y * y)


@pyflamegpu.agent_function
def human_perception_resource_locations(
    message_in: pyflamegpu.MessageSpatial2D, message_out: pyflamegpu.MessageNone
):
    agent_x = pyflamegpu.getVariableInt("x")
    agent_y = pyflamegpu.getVariableInt("y")
    # should be math.inf, but traspiling fails: fix would be to use std::numeric_limits<double>::max()
    closest_resource = 1.7976931348623157e308
    closest_resource_x = 0.0
    closest_resource_y = 0.0
    for resource in message_in.wrap(agent_x, agent_y):
        resource_x = resource.getVariableInt("x")
        resource_y = resource.getVariableInt("y")
        d = vec2Length(agent_x - resource_x, agent_y - resource_y)
        if d < closest_resource:
            closest_resource = d
            closest_resource_x = resource_x
            closest_resource_y = resource_y
    pyflamegpu.setVariableFloat("closest_resource", closest_resource)
    pyflamegpu.setVariableFloat("closest_resource_x", closest_resource_x)
    pyflamegpu.setVariableFloat("closest_resource_y", closest_resource_y)


@pyflamegpu.agent_function
def human_perception_human_locations(
    message_in: pyflamegpu.MessageSpatial2D, message_out: pyflamegpu.MessageNone
):
    id = pyflamegpu.getID()
    human_x = pyflamegpu.getVariableInt("x")
    human_y = pyflamegpu.getVariableInt("y")
    close_humans = 0
    for human in message_in.wrap(human_x, human_y):
        if human.getVariableInt("id") == id:
            continue
        other_human_x = human.getVariableInt("x")
        other_human_y = human.getVariableInt("y")
        if human_x == other_human_x and human_y == other_human_y:
            close_humans += 1
    if close_humans >= pyflamegpu.environment.getPropertyInt("N_HUMANS_CROWDED"):
        ap = pyflamegpu.getVariableFloat("actionpotential")
        ap -= pyflamegpu.environment.getPropertyFloat("AP_REDUCTION_BY_CROWDING")
        pyflamegpu.setVariableFloat("actionpotential", ap)


@pyflamegpu.agent_function
def human_behavior(
    message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone
):
    ap = pyflamegpu.getVariableFloat("actionpotential")
    can_collect_resource = ap >= pyflamegpu.environment.getPropertyFloat(
        "AP_COLLECT_RESOURCE"
    )
    can_move = ap >= pyflamegpu.environment.getPropertyFloat("AP_MOVE")
    if not (can_move or can_collect_resource):
        pyflamegpu.setVariableFloat(
            "actionpotential",
            pyflamegpu.environment.getPropertyFloat("AP_PER_TICK_RESTING"),
        )
        return
    if can_collect_resource and pyflamegpu.getVariableFloat(
        "closest_resource"
    ) < pyflamegpu.environment.getPropertyFloat("RESOURCE_COLLECTION_RANGE"):
        ap -= pyflamegpu.environment.getPropertyFloat("AP_COLLECT_RESOURCE")
        pyflamegpu.setVariableFloat("actionpotential", ap)
        resources = pyflamegpu.getVariableInt("resources")
        resources += 1
        pyflamegpu.setVariableInt("resources", resources)
        return
    if can_move and pyflamegpu.getVariableFloat(
        "closest_resource"
    ) <= pyflamegpu.environment.getPropertyFloat("HUMAN_MOVE_RANGE"):
        agent_x = pyflamegpu.getVariableInt("x")
        agent_y = pyflamegpu.getVariableInt("y")
        dx = math.abs(agent_x - pyflamegpu.getVariableFloat("closest_resource_x"))
        dy = math.abs(agent_y - pyflamegpu.getVariableFloat("closest_resource_y"))
        if dx > dy:
            x = (
                agent_x
                + (pyflamegpu.getVariableFloat("closest_resource_x") - agent_x) / dx
            )
            pyflamegpu.setVariableInt("x", x)
        else:
            y = (
                agent_y
                + (pyflamegpu.getVariableFloat("closest_resource_y") - agent_y) / dy
            )
            pyflamegpu.setVariableInt("y", y)
        ap -= pyflamegpu.environment.getPropertyFloat("AP_MOVE")
        pyflamegpu.setVariableFloat("actionpotential", ap)
    return pyflamegpu.ALIVE


@pyflamegpu.agent_function
def output_location(
    message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageSpatial2D
):
    message_out.setVariableInt("id", pyflamegpu.getID())
    message_out.setVariableInt("x", pyflamegpu.getVariableInt("x"))
    message_out.setVariableInt("y", pyflamegpu.getVariableInt("y"))
    return pyflamegpu.ALIVE


def make_human(model):
    human = model.newAgent("human")
    # properties of a human agent
    human.newVariableInt("x")
    human.newVariableInt("y")
    human.newVariableInt("resources")
    human.newVariableFloat("actionpotential")
    # passing data between agent_functions
    human.newVariableFloat("closest_resource")
    human.newVariableFloat("closest_resource_x")
    human.newVariableFloat("closest_resource_y")
    return human


def make_resource(model):
    resource = model.newAgent("resource")
    resource.newVariableInt("x")
    resource.newVariableInt("y")
    resource.newVariableInt("type")
    return resource


# debugging only
CUDA_human_behavior = """
FLAMEGPU_DEVICE_FUNCTION float vec2Length(int x, int y) { return sqrtf(((x * x) + (y * y))); }

FLAMEGPU_AGENT_FUNCTION(human_behavior, flamegpu::MessageNone, flamegpu::MessageNone) {
    auto ap = FLAMEGPU->getVariable<float>("actionpotential");
    auto can_collect_resource =
        ap >= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
    auto can_move = ap >= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
    if ((!(can_move || can_collect_resource))) {
        FLAMEGPU->setVariable<float>(
            "actionpotential", FLAMEGPU->environment.getProperty<float>("AP_PER_TICK_RESTING"));
        return;
    }
    if ((can_collect_resource &&
         FLAMEGPU->getVariable<float>("closest_resource") <
             FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE"))) {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
        FLAMEGPU->setVariable<float>("actionpotential", ap);
        auto resources = FLAMEGPU->getVariable<int>("resources");
        resources += 1;
        FLAMEGPU->setVariable<int>("resources", resources);
        return;
    }
    if ((can_move && FLAMEGPU->getVariable<float>("closest_resource") <=
                         FLAMEGPU->environment.getProperty<float>("HUMAN_MOVE_RANGE"))) {
        auto agent_x = FLAMEGPU->getVariable<int>("x");
        auto agent_y = FLAMEGPU->getVariable<int>("y");
        auto dx = abs((agent_x - FLAMEGPU->getVariable<float>("closest_resource_x")));
        auto dy = abs((agent_y - FLAMEGPU->getVariable<float>("closest_resource_y")));
        if (dx > dy) {
            auto x =
                (agent_x + ((FLAMEGPU->getVariable<float>("closest_resource_x") - agent_x) / dx));
            FLAMEGPU->setVariable<int>("x", x);
        } else {
            auto y =
                (agent_y + ((FLAMEGPU->getVariable<float>("closest_resource_y") - agent_y) / dy));
            FLAMEGPU->setVariable<int>("y", y);
        }
        ap -= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
        FLAMEGPU->setVariable<float>("actionpotential", ap);
    }
    return flamegpu::ALIVE;
}
"""
CUDA_human_perception_human_locations = """
FLAMEGPU_AGENT_FUNCTION(human_perception_human_locations, flamegpu::MessageSpatial2D,
                        flamegpu::MessageNone) {
    auto id = FLAMEGPU->getID();
    auto human_x = FLAMEGPU->getVariable<int>("x");
    auto human_y = FLAMEGPU->getVariable<int>("y");
    auto close_humans = 0;
    for (const auto &human : FLAMEGPU->message_in.wrap(human_x, human_y)) {
        if (human.getVariable<int>("id") == id) {
            continue;
        }
        printf("human[%02d] found human[%02d]\\n", id, human.getVariable<int>("id"));
        auto other_human_x = human.getVariable<int>("x");
        auto other_human_y = human.getVariable<int>("y");
        if ((human_x == other_human_x && human_y == other_human_y)) {
            close_humans += 1;
        }
    }
    printf("found %d close humans, >= %u\\n", close_humans,
           FLAMEGPU->environment.getProperty<int>("N_HUMANS_CROWDED"));
    if (close_humans >= FLAMEGPU->environment.getProperty<int>("N_HUMANS_CROWDED")) {
        auto ap = FLAMEGPU->getVariable<float>("actionpotential");
        ap -= FLAMEGPU->environment.getProperty<float>("AP_REDUCTION_BY_CROWDING");
        printf("ap = %f\\n", ap);
        FLAMEGPU->setVariable<float>("actionpotential", ap);
    }
}
"""


def vprint(*args, **kwargs):
    if "-v" in sys.argv:
        print(*args, **kwargs)


def make_simulation():
    ctx = ostruct.OpenStruct()
    model = pyflamegpu.ModelDescription("test_human_behavior")
    env = model.Environment()
    for key in C:
        if key[0] == "_":
            continue
        val = C[key]
        if type(val) == float:
            vprint(f"env[{key},float] = {val}")
            env.newPropertyFloat(key, val)
        elif type(val) == int:
            vprint(f"env[{key},int] = {val}")
            env.newPropertyInt(key, val)
        else:
            raise RuntimeError("unknown environment variable type")

    def make_location_message(model, name):
        message = model.newMessageSpatial2D(name)
        # XXX: setRadius: if not divided by 2, messages wrap around the borders and occur multiple times
        message.setRadius(C.ENV_MAX / 2)
        message.setMin(0, 0)
        message.setMax(C.ENV_MAX, C.ENV_MAX)
        message.newVariableID("id")

    make_location_message(model, "resource_location")
    make_location_message(model, "human_location")
    ctx.human = make_human(model)
    ctx.resource = make_resource(model)

    # layer 1: location message output
    resource_output_location_transpiled = pyflamegpu.codegen.translate(output_location)
    resource_output_location_description = ctx.resource.newRTCFunction(
        "ouput_location", resource_output_location_transpiled
    )
    resource_output_location_description.setMessageOutput("resource_location")

    human_output_location_transpiled = pyflamegpu.codegen.translate(output_location)
    human_output_location_description = ctx.human.newRTCFunction(
        "ouput_location", human_output_location_transpiled
    )
    human_output_location_description.setMessageOutput("human_location")
    # layer 2: perception
    human_perception_resource_locations_transpiled = pyflamegpu.codegen.translate(
        human_perception_resource_locations
    )
    human_perception_resource_locations_description = ctx.human.newRTCFunction(
        "human_perception_resource_locations",
        human_perception_resource_locations_transpiled,
    )
    human_perception_resource_locations_description.setMessageInput("resource_location")
    human_behavior_transpiled = pyflamegpu.codegen.translate(human_behavior)
    human_perception_human_locations_transpiled = pyflamegpu.codegen.translate(
        human_perception_human_locations
    )
    # human_perception_human_locations_transpiled = CUDA_human_perception_human_locations
    human_perception_human_locations_description = ctx.human.newRTCFunction(
        "human_perception_human_locations",
        human_perception_human_locations_transpiled,
    )
    human_perception_human_locations_description.setMessageInput("human_location")
    # layer 3: behavior
    # human_behavior_transpiled = CUDA_human_behavior
    human_behavior_description = ctx.human.newRTCFunction(
        "human_behavior", human_behavior_transpiled
    )
    vprint(
        f"resource_output_location_transpiled:\n'''{resource_output_location_transpiled}'''"
    )
    vprint(
        f"human_perception_resource_locations_transpiled:\n'''{human_perception_resource_locations_transpiled}'''"
    )
    vprint(
        f"human_perception_human_locations_transpiled:\n'''{human_perception_human_locations_transpiled}'''"
    )
    vprint(f"human_behavior_transpiled:\n'''{human_behavior_transpiled}'''")

    # Identify the root of execution
    # model.addExecutionRoot(resource_output_location_description)
    l1 = model.newLayer("layer 1: location message output")
    l1.addAgentFunction(resource_output_location_description)
    l1.addAgentFunction(human_output_location_description)
    l2 = model.newLayer("layer 2: perception")
    l2.addAgentFunction(human_perception_resource_locations_description)
    # FIXME: this should also be layer 2 function, but gives an error when added there
    model.newLayer("layer 2.1: FIXME").addAgentFunction(
        human_perception_human_locations_description
    )
    model.newLayer("layer 3: behavior").addAgentFunction(human_behavior_description)
    # Add the step function to the model.
    # step_validation_fn = step_validation()
    # model.addStepFunction(step_validation_fn)
    simulation = pyflamegpu.CUDASimulation(model)
    step_log = pyflamegpu.StepLoggingConfig(model)
    step_log.setFrequency(1)
    step_log.agent("human").logCount()
    step_log.agent("human").logSumInt("resources")
    simulation.setStepLog(step_log)
    return model, simulation, ctx


def main():
    model, simulation, ctx = make_simulation()

    vprint("starting with:", sys.argv)
    simulation.initialise(sys.argv)
    if not simulation.SimulationConfig().input_file:
        # Seed the host RNG using the cuda simulations' RNG
        random.seed(simulation.SimulationConfig().random_seed)
        humans = pyflamegpu.AgentVector(ctx.human, C.AGENT_COUNT)
        for h in humans:
            h.setVariableInt("x", int(random.uniform(0, C.ENV_MAX)))
            h.setVariableInt("y", int(random.uniform(0, C.ENV_MAX)))
        simulation.setPopulationData(humans)
    simulation.simulate()
    pyflamegpu.cleanup()  # Ensure profiling / memcheck work correctly


if __name__ == "__main__":
    main()
