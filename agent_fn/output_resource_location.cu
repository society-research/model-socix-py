#include "agent_fn/shared.cuh"

FLAMEGPU_AGENT_FUNCTION(resource_location, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    int amount = FLAMEGPU->getVariable<int>("amount");
    int regrowth_timer = FLAMEGPU->getVariable<int>("regrowth_timer");
    if (amount <= 0) {
        if (regrowth_timer ==
            FLAMEGPU->environment.getProperty<int>("RESOURCE_RESTORATION_TICKS")) {
            regrowth_timer = 0;
            amount = FLAMEGPU->environment.getProperty<int>("RESOURCE_DEPLETED_AFTER_COLLECTIONS");
        } else {
            regrowth_timer += 1;
        }
    }
    FLAMEGPU->setVariable<int>("regrowth_timer", regrowth_timer);
    FLAMEGPU->setVariable<int>("amount", amount);
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("x", FLAMEGPU->getVariable<int>("x"));
    FLAMEGPU->message_out.setVariable<int>("y", FLAMEGPU->getVariable<int>("y"));
    FLAMEGPU->message_out.setVariable<int>("type", FLAMEGPU->getVariable<int>("type"));
    FLAMEGPU->message_out.setVariable<int>("amount", amount);
}
