#include "agent_fn/shared.cuh"

// Note: must not increase function name length, since clang-format will create a line-break, which
// triggers a bug in the FLAMEGPU_AGENT_FUNCTION macro.
FLAMEGPU_AGENT_FUNCTION(resource_locations, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    int agent_x = FLAMEGPU->getVariable<int>("x");
    int agent_y = FLAMEGPU->getVariable<int>("y");
    float closest_resource_distance[N_RESOURCE_TYPES];
    int closest_resource_x[N_RESOURCE_TYPES];
    int closest_resource_y[N_RESOURCE_TYPES];
    for (int resource_type = 0; resource_type < N_RESOURCE_TYPES; resource_type++) {
        closest_resource_distance[resource_type] = FLT_MAX;
        closest_resource_x[resource_type] = 0;
        closest_resource_y[resource_type] = 0;
    }
    for (const auto &resource : FLAMEGPU->message_in.wrap(agent_x, agent_y)) {
        int resource_type = resource.getVariable<int>("type");
        int resource_x = resource.getVariable<int>("x");
        int resource_y = resource.getVariable<int>("y");
        float d = vec2Length((agent_x - resource_x), (agent_y - resource_y));
        if (d < closest_resource_distance[resource_type]) {
            closest_resource_distance[resource_type] = d;
            closest_resource_x[resource_type] = resource_x;
            closest_resource_y[resource_type] = resource_y;
        }
    }
    for (int resource_type = 0; resource_type < N_RESOURCE_TYPES; resource_type++) {
        FLAMEGPU->setVariable<float, 2>("closest_resource", resource_type,
                                        closest_resource_distance[resource_type]);
        FLAMEGPU->setVariable<int, 2>("closest_resource_x", resource_type,
                                      closest_resource_x[resource_type]);
        FLAMEGPU->setVariable<int, 2>("closest_resource_y", resource_type,
                                      closest_resource_y[resource_type]);
    }
}
