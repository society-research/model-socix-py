#include "agent_fn/function.cuh"

// not possible to include nvstd::function header, see
// https://github.com/FLAMEGPU/FLAMEGPU2/discussions/1199#discussioncomment-9146551
namespace Action {
enum Action {
    RandomWalk = 0,
    Rest = 1,
    CollectResource = 2,
    MoveToClosestResource = 3,
    EOF = 4,
};
}

FLAMEGPU_DEVICE_FUNCTION float vec2Length(int x, int y) { return sqrtf(((x * x) + (y * y))); }
FLAMEGPU_DEVICE_FUNCTION int findMax(int ar[], int len) {
    int max_index = len;
    int max = 0;
    for (int i = 0; i < len; i++) {
        // printf("[%02d] score[%d]=%d\n", FLAMEGPU->getID(), i, scores[i]);
        if (ar[i] > max) {
            max = ar[i];
            max_index = i;
        }
    }
    return max_index;
}
FLAMEGPU_DEVICE_FUNCTION void printAction(int a) {
    switch (a) {
    case Action::RandomWalk:
        printf("random_walk();\n");
        break;
    case Action::Rest:
        printf("rest();\n");
        break;
    case Action::CollectResource:
        printf("collect_resource();\n");
        break;
    case Action::MoveToClosestResource:
        printf("move_to_closest_resource();\n");
        break;
    case Action::EOF:
    default:
        printf("Action::EOF");
        break;
    }
}

FLAMEGPU_AGENT_FUNCTION(human_behavior, flamegpu::MessageNone, flamegpu::MessageNone) {
    float ap = FLAMEGPU->getVariable<float>("actionpotential");
    int x = FLAMEGPU->getVariable<int>("x");
    int y = FLAMEGPU->getVariable<int>("y");
    int hunger = FLAMEGPU->getVariable<int>("hunger");
    int resources = FLAMEGPU->getVariable<int>("resources");

    auto random_walk = [&]() {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
        int d = 0;
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
        int max = FLAMEGPU->environment.getProperty<int>("GRID_SIZE");
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
    auto collect_resource = [&]() {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
        resources += 1;
    };
    auto move_to_closest_resource = [&]() {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
        float dx = abs((x - FLAMEGPU->getVariable<float>("closest_resource_x")));
        float dy = abs((y - FLAMEGPU->getVariable<float>("closest_resource_y")));
        if (dx > dy) {
            x = (x + ((FLAMEGPU->getVariable<float>("closest_resource_x") - x) / dx));
        } else {
            y = (y + ((FLAMEGPU->getVariable<float>("closest_resource_y") - y) / dy));
        }
    };
    auto rest = [&]() {
        ap += FLAMEGPU->environment.getProperty<float>("AP_PER_TICK_RESTING");
        if (FLAMEGPU->getVariable<int>("is_crowded") == 1) {
            // remove the AP reduction by crowding in case of resting
            ap += FLAMEGPU->environment.getProperty<float>("AP_REDUCTION_BY_CROWDING");
        }
    };
    {
        // changes due to self-perception
        if (FLAMEGPU->getVariable<int>("is_crowded") == 1) {
            ap -= FLAMEGPU->environment.getProperty<float>("AP_REDUCTION_BY_CROWDING");
        }
        hunger += FLAMEGPU->environment.getProperty<int>("HUNGER_PER_TICK");
        if (hunger >= FLAMEGPU->environment.getProperty<int>("HUNGER_STARVED_TO_DEATH")) {
            return flamegpu::DEAD;
        }
        // XXX: strictly speaking food consumption is behavior and should be
        // scored below before executed
        if (resources != 0 &&
            hunger > FLAMEGPU->environment.getProperty<int>("HUNGER_TO_TRIGGER_CONSUMPTION")) {
            resources -= 1;
            hunger -= FLAMEGPU->environment.getProperty<int>("HUNGER_PER_RESOURCE_CONSUMPTION");
        }
    }
    int scores[Action::EOF];
    memset(&scores, 0, Action::EOF * sizeof(int));
    scores[Action::Rest] = 1;
    bool can_collect_resource =
        ap >= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
    bool can_move = ap >= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
    if (!(can_move || can_collect_resource)) {
        scores[Action::Rest] = 5;
    }
    if (can_move && FLAMEGPU->getVariable<int>("is_crowded") == 1) {
        scores[Action::RandomWalk] = 10;
    }
    float closest_resource = FLAMEGPU->getVariable<float>("closest_resource");
    if (can_collect_resource &&
        closest_resource <= FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE")) {
        scores[Action::CollectResource] = 10;
    }
    if (can_move &&
        closest_resource > FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE")) {
        scores[Action::MoveToClosestResource] =
            int(10 - closest_resource * FLAMEGPU->environment.getProperty<float>(
                                            "SCORE_REDUCTION_PER_TILE_DISTANCE"));
    }
    int selected_action = findMax(scores, Action::EOF);
    // printAction(selected_action);
    switch (selected_action) {
    case Action::RandomWalk:
        random_walk();
        break;
    case Action::Rest:
        rest();
        break;
    case Action::CollectResource:
        collect_resource();
        break;
    case Action::MoveToClosestResource:
        move_to_closest_resource();
        break;
    case Action::EOF:
    default:
        printf("[BUG] must not happen\n");
        break;
    }
    FLAMEGPU->setVariable<int>("x", x);
    FLAMEGPU->setVariable<int>("y", y);
    FLAMEGPU->setVariable<float>("actionpotential", ap);
    FLAMEGPU->setVariable<int>("hunger", hunger);
    FLAMEGPU->setVariable<int>("resources", resources);

    return flamegpu::ALIVE;
}
