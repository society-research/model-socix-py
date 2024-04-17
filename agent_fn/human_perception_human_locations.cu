FLAMEGPU_DEVICE_FUNCTION float vec2Length(int x, int y) { return sqrtf(((x * x) + (y * y))); }

FLAMEGPU_AGENT_FUNCTION(human_perception_human_locations, flamegpu::MessageSpatial2D,
                        flamegpu::MessageNone) {
    auto id = FLAMEGPU->getID();
    auto human_x = FLAMEGPU->getVariable<int>("x");
    auto human_y = FLAMEGPU->getVariable<int>("y");
    auto close_humans = 0;
    for (const auto &human : FLAMEGPU->message_in.wrap(human_x, human_y)) {
        if (human.getVariable<int>("id") == id) {
            continue;
        }
        auto other_human_x = human.getVariable<int>("x");
        auto other_human_y = human.getVariable<int>("y");
        if ((human_x == other_human_x && human_y == other_human_y)) {
            close_humans += 1;
        }
    }
    if (close_humans >= FLAMEGPU->environment.getProperty<int>("N_HUMANS_CROWDED")) {
        FLAMEGPU->setVariable<int>("is_crowded", 1);
    } else {
        FLAMEGPU->setVariable<int>("is_crowded", 0);
    }
}
