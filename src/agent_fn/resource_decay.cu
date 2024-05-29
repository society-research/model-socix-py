#include "agent_fn/shared.cuh"

FLAMEGPU_AGENT_FUNCTION(resource_decay, flamegpu::MessageBucket, flamegpu::MessageNone) {
    flamegpu::id_t resource_id = FLAMEGPU->getID();
    int amount = FLAMEGPU->getVariable<int>("amount");
    for (const auto &message : FLAMEGPU->message_in(resource_id)) {
        int amount_collected = message.getVariable<int>("amount");
        amount -= amount_collected;
    }
    // TODO: should avoid negative amount by making resource collection a 2-tick protocol
    if (amount <= 0) {
        amount = 0;
    }
    FLAMEGPU->setVariable<int>("amount", amount);
    return flamegpu::ALIVE;
}
