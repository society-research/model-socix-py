#include "agent_fn/shared.cuh"

FLAMEGPU_AGENT_FUNCTION(output_location, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    // printf("{r[%d]@[%d,%d]}", FLAMEGPU->getVariable<int>("type"),
    // FLAMEGPU->getVariable<int>("x"), FLAMEGPU->getVariable<int>("y"));
    FLAMEGPU->message_out.setVariable<int>("x", FLAMEGPU->getVariable<int>("x"));
    FLAMEGPU->message_out.setVariable<int>("y", FLAMEGPU->getVariable<int>("y"));
    FLAMEGPU->message_out.setVariable<int>("type", FLAMEGPU->getVariable<int>("type"));
}
