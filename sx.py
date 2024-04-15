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
    HUMAN_MOVE_RANGE=10.0,
    # default amount of action potential (AP)
    AP_DEFAULT=1.0,
    # AP needed to collect a resource
    AP_COLLECT_RESOURCE=0.1,
    # AP needed to move a block
    AP_MOVE=0.05,
    # required sleep per night in hours
    SLEEP_REQUIRED_PER_NIGHT=8,
)
C.ENV_MAX = math.floor(sqbrt(C.AGENT_COUNT))
C.AP_PER_TICK_RESTING = C.AP_DEFAULT / C.SLEEP_REQUIRED_PER_NIGHT


@pyflamegpu.device_function
def vec2Length(x: int, y: int) -> float:
    return math.sqrtf(x * x + y * y)


@pyflamegpu.agent_function
def human_behavior(
    message_in: pyflamegpu.MessageSpatial2D, message_out: pyflamegpu.MessageNone
):
    # TODO: this should be agent_function_condition
    agent_x = pyflamegpu.getVariableInt("x")
    agent_y = pyflamegpu.getVariableInt("y")
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

    # TODO: extract this into a separate agent_function
    # should be sys.float_info.max (but traspiling is not supported)
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

    if (
        can_collect_resource
        and closest_resource
        < pyflamegpu.environment.getPropertyFloat("RESOURCE_COLLECTION_RANGE")
    ):
        ap -= pyflamegpu.environment.getPropertyFloat("AP_COLLECT_RESOURCE")
        pyflamegpu.setVariableFloat("actionpotential", ap)
        resources = pyflamegpu.getVariableInt("resources")
        resources += 1
        pyflamegpu.setVariableInt("resources", resources)
        return
    if can_move and closest_resource <= pyflamegpu.environment.getPropertyFloat(
        "HUMAN_MOVE_RANGE"
    ):
        dx = math.abs(agent_x - closest_resource_x)
        dy = math.abs(agent_y - closest_resource_y)
        if dx > dy:
            x = agent_x + (closest_resource_x - agent_x) / dx
            pyflamegpu.setVariableInt("x", x)
        else:
            y = agent_y + (closest_resource_y - agent_y) / dy
            pyflamegpu.setVariableInt("y", y)
        ap -= pyflamegpu.environment.getPropertyFloat("AP_MOVE")
        pyflamegpu.setVariableFloat("actionpotential", ap)
    return pyflamegpu.ALIVE


@pyflamegpu.agent_function
def output_location(
    message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageSpatial2D
):
    message_out.setVariableInt("id", pyflamegpu.getID())
    message_out.setVariableFloat("x", pyflamegpu.getVariableFloat("x"))
    message_out.setVariableFloat("y", pyflamegpu.getVariableFloat("y"))
    return pyflamegpu.ALIVE


def make_human(model):
    human = model.newAgent("human")
    human.newVariableInt("x")
    human.newVariableInt("y")
    human.newVariableInt("resources")
    human.newVariableFloat("actionpotential")
    return human


def make_resource(model):
    resource = model.newAgent("resource")
    resource.newVariableInt("x")
    resource.newVariableInt("y")
    resource.newVariableInt("type")
    return resource


# debugging only
CUDA_OUTPUT_LOCATION = """
FLAMEGPU_AGENT_FUNCTION(output_location, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    printf("called output_location\\n");
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}
"""
CUDA_human_behavior = """
FLAMEGPU_DEVICE_FUNCTION float vec2Length(int x, int y) { return sqrtf(((x * x) + (y * y))); }

FLAMEGPU_AGENT_FUNCTION(human_behavior, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    auto agent_x = FLAMEGPU->getVariable<int>("x");
    auto agent_y = FLAMEGPU->getVariable<int>("y");
    for (const auto &message : FLAMEGPU->message_in.wrap(agent_x, agent_y)) {
        auto message_x = message.getVariable<int>("x");
        auto message_y = message.getVariable<int>("y");
        auto d = vec2Length((agent_x - message_x), (agent_y - message_y));
        if (d < FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE")) {
            auto ap = FLAMEGPU->getVariable<float>("actionpotential");
            ap -= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
            printf("setting ap=%f\\n", ap);
            FLAMEGPU->setVariable<float>("actionpotential", ap);
            auto resources = FLAMEGPU->getVariable<int>("resources");
            resources += 1;
            FLAMEGPU->setVariable<int>("resources", resources);
            return;
        }
        if (d <= FLAMEGPU->environment.getProperty<float>("HUMAN_MOVE_RANGE")) {
            auto dx = abs((agent_x - message_x));
            auto dy = abs((agent_y - message_y));
            if (dx > dy) {
                auto x = (agent_x + ((message_x - agent_x) / dx));
                FLAMEGPU->setVariable<int>("x", x);
            } else {
                auto y = (agent_y + ((message_y - agent_y) / dy));
                FLAMEGPU->setVariable<int>("y", y);
            }
            break;
        }
    }
    return flamegpu::ALIVE;
}
"""


def make_simulation():
    ctx = ostruct.OpenStruct()
    model = pyflamegpu.ModelDescription("test_human_behavior")
    env = model.Environment()
    for key in C:
        if key[0] == "_":
            continue
        env.newPropertyFloat(key, C[key])
    message = model.newMessageSpatial2D("resource_location")
    message.setRadius(C.ENV_MAX)
    message.setMin(0, 0)
    message.setMax(C.ENV_MAX, C.ENV_MAX)
    message.newVariableID("id")
    ctx.human = make_human(model)
    ctx.resource = make_resource(model)
    output_location_transpiled = pyflamegpu.codegen.translate(output_location)
    # output_location_transpiled = CUDA_OUTPUT_LOCATION
    output_location_description = ctx.resource.newRTCFunction(
        "ouput_location", output_location_transpiled
    )
    output_location_description.setMessageOutput("resource_location")
    human_behavior_transpiled = pyflamegpu.codegen.translate(human_behavior)
    # human_behavior_transpiled = CUDA_human_behavior
    human_behavior_description = ctx.human.newRTCFunction(
        "human_behavior", human_behavior_transpiled
    )
    human_behavior_description.setMessageInput("resource_location")
    if "-v" in sys.argv:
        print(f"output_location_transpiled:\n'''{output_location_transpiled}'''")
        print(f"human_behavior_transpiled:\n'''{human_behavior_transpiled}'''")
    # Identify the root of execution
    # model.addExecutionRoot(output_location_description)
    model.newLayer("layer 1").addAgentFunction(output_location_description)
    model.newLayer("layer 2").addAgentFunction(human_behavior_description)
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

    print("starting with:", sys.argv)
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
