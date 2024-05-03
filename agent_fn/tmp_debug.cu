#include "agent_fn/shared.cuh"

FLAMEGPU_AGENT_FUNCTION(tmp_debug, flamegpu::MessageNone, flamegpu::MessageNone) {
    //printf("XXX<%d>[%d,%d]", FLAMEGPU->getVariable<int>("type"), FLAMEGPU->getVariable<int>("x"),
    //       FLAMEGPU->getVariable<int>("y"));
}
