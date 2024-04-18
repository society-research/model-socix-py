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
    return pyflamegpu.ALIVE


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
        pyflamegpu.setVariableInt("is_crowded", 1)
    else:
        pyflamegpu.setVariableInt("is_crowded", 0)


@pyflamegpu.agent_function
def human_behavior(
    message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone
):
    ap = pyflamegpu.getVariableFloat("actionpotential")
    x = pyflamegpu.getVariableInt("x")
    y = pyflamegpu.getVariableInt("y")
    if pyflamegpu.getVariableInt("is_crowded") == 1:
        ap -= pyflamegpu.environment.getPropertyFloat("AP_REDUCTION_BY_CROWDING")
    can_collect_resource = ap >= pyflamegpu.environment.getPropertyFloat(
        "AP_COLLECT_RESOURCE"
    )
    can_move = ap >= pyflamegpu.environment.getPropertyFloat("AP_MOVE")
    if not (can_move or can_collect_resource):
        ap += pyflamegpu.environment.getPropertyFloat("AP_PER_TICK_RESTING")
    elif can_move and pyflamegpu.getVariableInt("is_crowded") == 1:
        ap -= pyflamegpu.environment.getPropertyFloat("AP_MOVE")
        d = 0
        if pyflamegpu.random.uniformInt(0, 1) == 0:
            d = 1
        else:
            d = -1
        if pyflamegpu.random.uniformInt(0, 1) == 0:
            x += d
        else:
            y += d
        # wrap around
        max = pyflamegpu.environment.getPropertyInt("GRID_SIZE")
        if x < 0:
            x = max
        elif y < 0:
            y = max
        elif x == max:
            x = 0
        elif y == max:
            y = 0
        pyflamegpu.setVariableInt("x", x)
        pyflamegpu.setVariableInt("y", y)
    elif can_collect_resource and pyflamegpu.getVariableFloat(
        "closest_resource"
    ) < pyflamegpu.environment.getPropertyFloat("RESOURCE_COLLECTION_RANGE"):
        ap -= pyflamegpu.environment.getPropertyFloat("AP_COLLECT_RESOURCE")
        resources = pyflamegpu.getVariableInt("resources")
        resources += 1
        pyflamegpu.setVariableInt("resources", resources)
    elif can_move and pyflamegpu.getVariableFloat(
        "closest_resource"
    ) <= pyflamegpu.environment.getPropertyFloat("HUMAN_MOVE_RANGE"):
        ap -= pyflamegpu.environment.getPropertyFloat("AP_MOVE")
        dx = math.abs(x - pyflamegpu.getVariableFloat("closest_resource_x"))
        dy = math.abs(y - pyflamegpu.getVariableFloat("closest_resource_y"))
        if dx > dy:
            new_x = x + (pyflamegpu.getVariableFloat("closest_resource_x") - x) / dx
            pyflamegpu.setVariableInt("x", new_x)
        else:
            new_y = y + (pyflamegpu.getVariableFloat("closest_resource_y") - y) / dy
            pyflamegpu.setVariableInt("y", new_y)
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
    human.newVariableInt("is_crowded")
    return human


def make_resource(model):
    resource = model.newAgent("resource")
    resource.newVariableInt("x")
    resource.newVariableInt("y")
    resource.newVariableInt("type")
    return resource


def vprint(*args, **kwargs):
    if "-v" in sys.argv:
        print(*args, **kwargs)


def make_simulation(grid_size=10):
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
    env.newPropertyInt("GRID_SIZE", grid_size)

    def make_location_message(model, name):
        message = model.newMessageSpatial2D(name)
        # XXX: setRadius: if not divided by 2, messages wrap around the borders and occur multiple times
        message.setRadius(grid_size / 2)
        message.setMin(0, 0)
        message.setMax(grid_size, grid_size)
        message.newVariableID("id")

    make_location_message(model, "resource_location")
    make_location_message(model, "human_location")
    ctx.human = make_human(model)
    ctx.resource = make_resource(model)

    def make_agent_function(agent, name, py_fn=None, cuda_fn=None, cuda_fn_file=None):
        "Either `py_fn` or `cuda_fn` must be passed"
        if sum([1 for i in [py_fn, cuda_fn, cuda_fn_file] if i]) > 1:
            raise RuntimeError("use one argument only to pass the agent function")
        if cuda_fn_file is not None:
            with open(cuda_fn_file) as fd:
                cuda_fn = fd.read()
        if py_fn is not None:
            cuda_fn = pyflamegpu.codegen.translate(py_fn)
        description = agent.newRTCFunction(name, cuda_fn)
        vprint(f"{name}: '''\n{cuda_fn}'''")
        return description

    # layer 1: location message output
    resource_output_location_description = make_agent_function(
        ctx.resource, "output_location", py_fn=output_location
    )
    resource_output_location_description.setMessageOutput("resource_location")
    human_output_location_description = make_agent_function(
        ctx.human, "output_location", py_fn=output_location
    )
    human_output_location_description.setMessageOutput("human_location")
    # layer 2: perception
    human_perception_resource_locations_description = make_agent_function(
        ctx.human,
        "human_perception_resource_locations",
        py_fn=human_perception_resource_locations,
    )
    human_perception_resource_locations_description.setMessageInput("resource_location")
    human_perception_human_locations_description = make_agent_function(
        ctx.human,
        "human_perception_human_locations",
        py_fn=human_perception_human_locations,
    )
    human_perception_human_locations_description.setMessageInput("human_location")
    # layer 3: behavior
    human_behavior_description = make_agent_function(
        ctx.human,
        "human_behavior",
        #py_fn=human_behavior,
        cuda_fn_file="agent_fn/human_behavior.cu"
    )

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
    env_max = 100
    model, simulation, ctx = make_simulation(grid_size=env_max)

    vprint("starting with:", sys.argv)
    simulation.initialise(sys.argv)
    if not simulation.SimulationConfig().input_file:
        # Seed the host RNG using the cuda simulations' RNG
        random.seed(simulation.SimulationConfig().random_seed)
        humans = pyflamegpu.AgentVector(ctx.human, C.AGENT_COUNT)
        for h in humans:
            h.setVariableInt("x", int(random.uniform(0, env_max)))
            h.setVariableInt("y", int(random.uniform(0, env_max)))
        simulation.setPopulationData(humans)
    simulation.simulate()
    pyflamegpu.cleanup()  # Ensure profiling / memcheck work correctly


if __name__ == "__main__":
    main()
