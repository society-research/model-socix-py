import sys
import os
import random
import ostruct
import pyflamegpu
import pyflamegpu.codegen


def sqbrt(x):
    root = abs(x) ** (1 / 2)
    return root if x >= 0 else -root


C = ostruct.OpenStruct(
    AGENT_COUNT=64,
    # amount of different resource types
    N_RESOURCE_TYPES=2,
    # a resource is depleted after this amount of collections, and needs some
    # time to regenerate
    RESOURCE_DEPLETED_AFTER_COLLECTIONS=5,
    # number of ticks after which a depleted resource is available again
    RESOURCE_RESTORATION_TICKS=2,
    # default amount of action potential (AP)
    AP_DEFAULT=1.0,
    # AP needed to collect a resource
    AP_COLLECT_RESOURCE=0.1,
    # AP needed to move a block
    AP_MOVE=0.05,
    # euclidean distance at which resource is in range for collection
    RESOURCE_COLLECTION_RANGE=3.0,
    # after being in possession of this amount of resources a human, will do something else
    TARGET_RESOURCE_AMOUNT=5,
    # required sleep per night in hours
    SLEEP_REQUIRED_PER_NIGHT=8,
    # amount of humans in a single tile, after which humans feel crowded
    # (reduced AP)
    N_HUMANS_CROWDED=10,
    # distance * SCORE_REDUCTION_PER_TILE_DISTANCE is subtracted from the score
    # calculation when a human agent tries to evaluate its actions
    SCORE_REDUCTION_PER_TILE_DISTANCE=0.1,
    # increase in hunger per tick
    HUNGER_PER_TICK=1,
    # amount of hunger a single consumed resource restores
    HUNGER_PER_RESOURCE_CONSUMPTION=8,
    # amount of hunger when a human starves to death
    HUNGER_STARVED_TO_DEATH=300,
)
# after this amount of hunger a human chooses to eat if possible
C.HUNGER_TO_TRIGGER_CONSUMPTION = C.HUNGER_PER_RESOURCE_CONSUMPTION
# Restored AP per tick spend resting.
C.AP_PER_TICK_RESTING = C.AP_DEFAULT / C.SLEEP_REQUIRED_PER_NIGHT
# AP reduction caused by crowding, see @N_HUMANS_CROWDED
# Note: humans should be able to move, even with reduced AP by crowding
C.AP_REDUCTION_BY_CROWDING = C.AP_DEFAULT / 10


@pyflamegpu.agent_function
def human_perception_human_locations(
    message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageNone
):
    id = pyflamegpu.getID()
    human_x = pyflamegpu.getVariableInt("x")
    human_y = pyflamegpu.getVariableInt("y")
    close_humans = 0
    for human in message_in:
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
    return pyflamegpu.ALIVE


@pyflamegpu.agent_function
def human_location(
    message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce
):
    message_out.setVariableInt("id", pyflamegpu.getID())
    message_out.setVariableInt("x", pyflamegpu.getVariableInt("x"))
    message_out.setVariableInt("y", pyflamegpu.getVariableInt("y"))


def make_human(model):
    human = model.newAgent("human")
    # properties of a human agent
    human.newVariableInt("x")
    human.newVariableInt("y")
    human.newVariableArrayInt("resources", C.N_RESOURCE_TYPES, [0, 0])
    human.newVariableFloat("actionpotential")
    human.newVariableInt("hunger")
    # passing data between agent_functions
    human.newVariableArrayFloat("closest_resource", C.N_RESOURCE_TYPES, [0, 0])
    human.newVariableArrayInt("closest_resource_x", C.N_RESOURCE_TYPES, [0, 0])
    human.newVariableArrayInt("closest_resource_y", C.N_RESOURCE_TYPES, [0, 0])
    human.newVariableArrayInt("closest_resource_id", C.N_RESOURCE_TYPES, [0, 0])
    human.newVariableInt("is_crowded")
    # analysis data
    human.newVariableArrayInt("ana_last_resource_location", 2, [-1, -1])
    return human


def make_resource(model):
    resource = model.newAgent("resource")
    resource.newVariableInt("x", 0)
    resource.newVariableInt("y", 0)
    resource.newVariableInt("type", 0)
    resource.newVariableInt("amount", C.RESOURCE_DEPLETED_AFTER_COLLECTIONS)
    resource.newVariableInt("regrowth_timer", 0)
    return resource


def vprint(*args, **kwargs):
    if "-vv" in sys.argv:
        print(*args, **kwargs)


def make_simulation(
    grid_size=10,
    max_resources=100,
) -> [pyflamegpu.ModelDescription, pyflamegpu.CUDASimulation, ostruct.OpenStruct]:
    """Create s FLAMEGPU simulation with actors & messages defined, but no
    created actors (agent vectors) set.

    Parameters:
        grid_size (int): border size of the 2D grid (square)
        max_resources (int): maximum amount of resources that you promise to
            add as agents! if adding more resource agents behavior is undefined

    Returns: modelDescription, CUDASimulation, constants
        modelDescription, CUDASimulation (flamegpu2 swig types)
        constants (openstruct): containing all model parameters
    """
    ctx = ostruct.OpenStruct()
    model = pyflamegpu.ModelDescription("socix")
    env = model.Environment()
    for key in C:
        if key[0] == "_":
            continue
        val = C[key]
        if type(val) is float:
            vprint(f"env[{key},float] = {val}")
            env.newPropertyFloat(key, val)
        elif type(val) is int:
            vprint(f"env[{key},int] = {val}")
            env.newPropertyInt(key, val)
        else:
            raise RuntimeError("unknown environment variable type")
    env.newPropertyInt("GRID_SIZE", grid_size)

    def make_location_message(model: pyflamegpu.ModelDescription, name: str):
        message = model.newMessageBruteForce(name)
        message.newVariableID("id")
        message.newVariableInt("x")
        message.newVariableInt("y")
        return message

    resource_msg = make_location_message(model, "resource_location")
    resource_msg.newVariableInt("type")
    resource_msg.newVariableInt("amount")
    make_location_message(model, "human_location")
    resource_collection_msg = model.newMessageBucket("resource_collection")
    resource_collection_msg.newVariableInt("amount")
    resource_collection_msg.setBounds(0, max_resources)
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
    cwd = os.path.dirname(os.path.abspath(__file__))
    resource_output_location_description = make_agent_function(
        ctx.resource,
        "output_location_and_type",
        cuda_fn_file=f"{cwd}/agent_fn/output_resource_location.cu",
    )
    resource_output_location_description.setMessageOutput("resource_location")
    human_output_location_description = make_agent_function(
        ctx.human, "human_location", py_fn=human_location
    )
    human_output_location_description.setMessageOutput("human_location")
    # layer 2: perception
    human_perception_resource_locations_description = make_agent_function(
        ctx.human,
        "human_perception_resource_locations",
        cuda_fn_file=f"{cwd}/agent_fn/human_perception_resource_locations.cu",
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
        cuda_fn_file=f"{cwd}/agent_fn/human_behavior.cu",
    )
    human_behavior_description.setMessageOutput("resource_collection")
    human_behavior_description.setAllowAgentDeath(True)
    # layer 4: environmental effects
    resource_decay_description = make_agent_function(
        ctx.resource,
        "resource_decay",
        cuda_fn_file=f"{cwd}/agent_fn/resource_decay.cu",
    )
    resource_decay_description.setMessageInput("resource_collection")
    # resource_decay_description.setAllowAgentDeath(True)

    # Identify the root of execution
    # model.addExecutionRoot(resource_output_location_description)
    l1 = model.newLayer("layer 1: location message output")
    l1.addAgentFunction(resource_output_location_description)
    l1.addAgentFunction(human_output_location_description)
    l2 = model.newLayer("layer 2.0: perception: resources")
    l2.addAgentFunction(human_perception_resource_locations_description)
    model.newLayer("layer 2.1: perception: humans").addAgentFunction(
        human_perception_human_locations_description
    )
    model.newLayer("layer 3: behavior").addAgentFunction(human_behavior_description)
    model.newLayer("layer 4: environmental effects").addAgentFunction(
        resource_decay_description
    )
    # Add the step function to the model.
    # step_validation_fn = step_validation()
    # model.addStepFunction(step_validation_fn)
    simulation = pyflamegpu.CUDASimulation(model)
    step_log = pyflamegpu.StepLoggingConfig(model)
    step_log.setFrequency(1)
    step_log.agent("human").logCount()
    # step_log.agent("human").logSumInt("resources") XXX: porbably not working with array of int
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
