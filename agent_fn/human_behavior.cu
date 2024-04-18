#include <type_traits>
#include <nvfunctional>
//#include <functional>

FLAMEGPU_DEVICE_FUNCTION float vec2Length(int x, int y) { return sqrtf(((x * x) + (y * y))); }

//class Action {
//  public:
//    // nvstd::function<void()> action;
//    Action(std::string action, float score) {
//        this->action = action;
//        this->score = score;
//    }
//    std::string action;
//    float score;
//};
struct Action {
  char id[10];
  float score;
};

FLAMEGPU_AGENT_FUNCTION(human_behavior, flamegpu::MessageNone, flamegpu::MessageNone) {
    auto ap = FLAMEGPU->getVariable<float>("actionpotential");
    auto x = FLAMEGPU->getVariable<int>("x");
    auto y = FLAMEGPU->getVariable<int>("y");
    auto random_walk = [&]() {
        auto d = 0;
        if (FLAMEGPU->random.uniform<int>(0, 1) == 0) {
            d = 1;
        } else {
            d = -1;
        }
        if (FLAMEGPU->random.uniform<int>(0, 1) == 0) {
            x += d;
        } else {
            y += d;
        }
        auto max = FLAMEGPU->environment.getProperty<int>("GRID_SIZE");
        if (x < 0) {
            x = max;
        } else if (y < 0) {
            y = max;
        } else if (x == max) {
            x = 0;
        } else if (y == max) {
            y = 0;
        }
    };
    Action actions[] = {
      {"ok", 0}
    };
    //Action actions[] = {
    //    Action("ok", 0),
    //};
    auto collect_resource = [&]() {
        auto resources = FLAMEGPU->getVariable<int>("resources");
        resources += 1;
        FLAMEGPU->setVariable<int>("resources", resources);
    };
    if (FLAMEGPU->getVariable<int>("is_crowded") == 1) {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_REDUCTION_BY_CROWDING");
    }
    bool can_collect_resource =
        ap >= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
    bool can_move = ap >= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
    if (!(can_move || can_collect_resource)) {
        ap += FLAMEGPU->environment.getProperty<float>("AP_PER_TICK_RESTING");
    } else if ((can_move && FLAMEGPU->getVariable<int>("is_crowded") == 1)) {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
        random_walk();
    } else if ((can_collect_resource &&
                FLAMEGPU->getVariable<float>("closest_resource") <
                    FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE"))) {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
        collect_resource();
    } else if ((can_move && FLAMEGPU->getVariable<float>("closest_resource") <=
                                FLAMEGPU->environment.getProperty<float>("HUMAN_MOVE_RANGE"))) {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
        auto dx = abs((x - FLAMEGPU->getVariable<float>("closest_resource_x")));
        auto dy = abs((y - FLAMEGPU->getVariable<float>("closest_resource_y")));
        if (dx > dy) {
            x = (x + ((FLAMEGPU->getVariable<float>("closest_resource_x") - x) / dx));
        } else {
            y = (y + ((FLAMEGPU->getVariable<float>("closest_resource_y") - y) / dy));
        }
    }
    FLAMEGPU->setVariable<int>("x", x);
    FLAMEGPU->setVariable<int>("y", y);
    FLAMEGPU->setVariable<float>("actionpotential", ap);
    return flamegpu::ALIVE;
}