import sys, random, math
import ostruct
from pyflamegpu import *
import pyflamegpu.codegen


def sqbrt(x):
    root = abs(x) ** (1 / 2)
    return root if x >= 0 else -root


AGENT_COUNT = 64
RESOURCE_COLLECTION_RANGE = 3.0
HUMAN_MOVE_RANGE = 10.0
ENV_MAX = math.floor(sqbrt(AGENT_COUNT))


@pyflamegpu.device_function
def vec2Length(x: int, y: int) -> float:
    return math.sqrtf(x * x + y * y)


@pyflamegpu.agent_function
def collect_resource(
    message_in: pyflamegpu.MessageSpatial2D, message_out: pyflamegpu.MessageNone
):
    agent_x = pyflamegpu.getVariableInt("x")
    agent_y = pyflamegpu.getVariableInt("y")
    for message in message_in.wrap(agent_x, agent_y):
        message_x = message.getVariableInt("x")
        message_y = message.getVariableInt("y")
        d = vec2Length(agent_x - message_x, agent_y - message_y)
        if d < pyflamegpu.environment.getPropertyFloat("RESOURCE_COLLECTION_RANGE"):
            resources = pyflamegpu.getVariableInt("resources")
            resources += 1
            pyflamegpu.setVariableInt("resources", resources)
            return
    for message in message_in.wrap(agent_x, agent_y):
        message_x = message.getVariableInt("x")
        message_y = message.getVariableInt("y")
        d = vec2Length(agent_x - message_x, agent_y - message_y)
        if d <= pyflamegpu.environment.getPropertyFloat("HUMAN_MOVE_RANGE"):
            dx = math.abs(agent_x - message_x)
            dy = math.abs(agent_y - message_y)
            if dx > dy:
                x = agent_x + (message_x - agent_x)/dx
                pyflamegpu.setVariableInt("x", x)
            else:
                y = agent_y + (message_y - agent_y)/dy
                pyflamegpu.setVariableInt("y", y)
            break
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
    human.newVariableInt("id")
    human.newVariableInt("x")
    human.newVariableInt("y")
    human.newVariableInt("resources")
    return human


def make_resource(model):
    resource = model.newAgent("resource")
    resource.newVariableInt("id")
    resource.newVariableInt("x")
    resource.newVariableInt("y")
    resource.newVariableInt("type")
    return resource


# debugging only
CUDA_OUTPUT_LOCATION = """
FLAMEGPU_AGENT_FUNCTION(output_location, flamegpu::MessageNone, flamegpu::MessageSpatial2D){
    printf("called output_location\\n");
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}
"""
CUDA_COLLECT_RESOURCE = """
FLAMEGPU_DEVICE_FUNCTION float vec2Length(int x, int y){
    return sqrtf(((x * x) + (y * y)));
}

FLAMEGPU_AGENT_FUNCTION(collect_resource, flamegpu::MessageSpatial2D, flamegpu::MessageNone){
    auto agent_x = FLAMEGPU->getVariable<int>("x");
    auto agent_y = FLAMEGPU->getVariable<int>("y");
    for (const auto& message : FLAMEGPU->message_in.wrap(agent_x, agent_y)){
        auto message_x = message.getVariable<int>("x");
        auto message_y = message.getVariable<int>("y");
        auto d = vec2Length((agent_x - message_x), (agent_y - message_y));
        if (d < FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE")){
            auto resources = FLAMEGPU->getVariable<int>("resources");
            resources += 1;
            FLAMEGPU->setVariable<int>("resources", resources);
            return;
        }
    }
    for (const auto& message : FLAMEGPU->message_in.wrap(agent_x, agent_y)){
        auto message_x = message.getVariable<int>("x");
        auto message_y = message.getVariable<int>("y");
        printf("found message at [%d,%d]\\n", message_x, message_y);
        auto d = vec2Length((agent_x - message_x), (agent_y - message_y));
        if (d <= FLAMEGPU->environment.getProperty<float>("HUMAN_MOVE_RANGE")){
            auto dx = abs((agent_x - message_x));
            auto dy = abs((agent_y - message_y));
            if (dx > dy){
                FLAMEGPU->setVariable<int>("x", 1);
            }
            else{
                int y = agent_y + ((message_y - agent_y) / dy);
                printf("move to y=%d (%d, %d, %d)\\n", y, message_y, agent_y, dy);
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
    model = pyflamegpu.ModelDescription("test_collect_resource")
    env = model.Environment()
    env.newPropertyFloat("RESOURCE_COLLECTION_RANGE", RESOURCE_COLLECTION_RANGE)
    env.newPropertyFloat("HUMAN_MOVE_RANGE", HUMAN_MOVE_RANGE)
    message = model.newMessageSpatial2D("resource_location")
    message.setRadius(ENV_MAX)
    message.setMin(0, 0)
    message.setMax(ENV_MAX, ENV_MAX)
    # message.newVariableInt("x")
    # message.newVariableInt("y")
    message.newVariableID("id")
    ctx.human = make_human(model)
    ctx.resource = make_resource(model)
    output_location_transpiled = pyflamegpu.codegen.translate(output_location)
    output_location_description = ctx.resource.newRTCFunction(
        "ouput_location", output_location_transpiled
    )
    output_location_description.setMessageOutput("resource_location")
    collect_resource_transpiled = pyflamegpu.codegen.translate(collect_resource)
    #collect_resource_transpiled = CUDA_COLLECT_RESOURCE
    collect_resource_description = ctx.human.newRTCFunction(
        "collect_resource", collect_resource_transpiled
    )
    collect_resource_description.setMessageInput("resource_location")
    if "-v" in sys.argv:
        print(f"output_location_transpiled:\n'''{output_location_transpiled}'''")
        print(f"collect_resource_transpiled:\n'''{collect_resource_transpiled}'''")
    ## Identify the root of execution
    # model.addExecutionRoot(output_location_description)
    model.newLayer("layer 1").addAgentFunction(output_location_description)
    model.newLayer("layer 2").addAgentFunction(collect_resource_description)
    ## Add the step function to the model.
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
        humans = pyflamegpu.AgentVector(ctx.human, AGENT_COUNT)
        for h in humans:
            h.setVariableInt("x", int(random.uniform(0, ENV_MAX)))
            h.setVariableInt("y", int(random.uniform(0, ENV_MAX)))
        simulation.setPopulationData(humans)
    simulation.simulate()
    pyflamegpu.cleanup()  # Ensure profiling / memcheck work correctly


if __name__ == "__main__":
    main()
