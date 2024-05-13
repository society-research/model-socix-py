#include "agent_fn/shared.cuh"

FLAMEGPU_AGENT_FUNCTION(resource_location, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("x", FLAMEGPU->getVariable<int>("x"));
    FLAMEGPU->message_out.setVariable<int>("y", FLAMEGPU->getVariable<int>("y"));
    FLAMEGPU->message_out.setVariable<int>("type", FLAMEGPU->getVariable<int>("type"));
    FLAMEGPU->message_out.setVariable<int>("amount", FLAMEGPU->getVariable<int>("amount"));
}
