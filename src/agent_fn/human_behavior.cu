#include "agent_fn/shared.cuh"

// not possible to include nvstd::function header, see
// https://github.com/FLAMEGPU/FLAMEGPU2/discussions/1199#discussioncomment-9146551
namespace Action {
enum Action {
    RandomWalk = 0,
    Rest,
    CollectResource0,
    CollectResource1,
    MoveToClosestResource0,
    MoveToClosestResource1,
    EOF,
};
}

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
FLAMEGPU_DEVICE_FUNCTION void printAction(int id, int a) {
    switch (a) {
    case Action::RandomWalk:
        printf("%d->random_walk();", id);
        break;
    case Action::Rest:
        printf("%d->rest();", id);
        break;
    case Action::CollectResource0:
        printf("%d->collect_resource(0);", id);
        break;
    case Action::CollectResource1:
        printf("%d->collect_resource(1);", id);
        break;
    case Action::MoveToClosestResource0:
        printf("%d->move_to_closest_resource(0);", id);
        break;
    case Action::MoveToClosestResource1:
        printf("%d->move_to_closest_resource(1);", id);
        break;
    case Action::EOF:
    default:
        printf("%d->Action::EOF;", id);
        break;
    }
    // printf("\n");
}

FLAMEGPU_DEVICE_FUNCTION float distance(int x, int y, int xo, int yo) {
    return sqrt(float((x - xo) * (x - xo) + (y - yo) * (y - yo)));
}

FLAMEGPU_AGENT_FUNCTION(human_behavior, flamegpu::MessageNone, flamegpu::MessageBucket) {
    float ap = FLAMEGPU->getVariable<float>("actionpotential");
    int x = FLAMEGPU->getVariable<int>("x");
    int y = FLAMEGPU->getVariable<int>("y");
    int hunger = FLAMEGPU->getVariable<int>("hunger");
    int resources[N_RESOURCE_TYPES];
    for (int resource_type = 0; resource_type < N_RESOURCE_TYPES; resource_type++) {
        resources[resource_type] =
            FLAMEGPU->getVariable<int, N_RESOURCE_TYPES>("resources", resource_type);
    }

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
    auto collect_resource = [&](int resource_type) {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_COLLECT_RESOURCE");
        resources[resource_type] += 1;
        // store analysis data
        int resource_x =
            FLAMEGPU->getVariable<int, N_RESOURCE_TYPES>("closest_resource_x", resource_type);
        int resource_y =
            FLAMEGPU->getVariable<int, N_RESOURCE_TYPES>("closest_resource_y", resource_type);
        auto resource_id = FLAMEGPU->getVariable<flamegpu::id_t, N_RESOURCE_TYPES>(
            "closest_resource_id", resource_type);
        // printf("collecting x=%d, y=%d\n", resource_x, resource_y);
        FLAMEGPU->setVariable<int, N_DIM>("ana_last_resource_location", 0, resource_x);
        FLAMEGPU->setVariable<int, N_DIM>("ana_last_resource_location", 1, resource_y);
        FLAMEGPU->message_out.setVariable<int>("amount", 1);
        FLAMEGPU->message_out.setKey(resource_id); // bucket of resource that is collected
    };
    auto move_to_closest_resource = [&](int resource_type) {
        ap -= FLAMEGPU->environment.getProperty<float>("AP_MOVE");
        int closest_x =
            FLAMEGPU->getVariable<int, N_RESOURCE_TYPES>("closest_resource_x", resource_type);
        int closest_y =
            FLAMEGPU->getVariable<int, N_RESOURCE_TYPES>("closest_resource_y", resource_type);
        int dx = abs((x - closest_x));
        int dy = abs((y - closest_y));
        int step_x = (closest_x - x) / dx;
        int step_y = (closest_y - y) / dy;
        float dist_after_x_step = vec2Dist(x + step_x, y, closest_x, closest_y);
        float dist_after_y_step = vec2Dist(x, y + step_y, closest_x, closest_y);
        if (dist_after_x_step < dist_after_y_step) {
            x += step_x;
        } else {
            y += step_y;
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
        // TODO(skep): strictly speaking food consumption is behavior and should be scored below
        // before being executed
        if (resources[0] != 0 && resources[1] != 0 &&
            hunger > FLAMEGPU->environment.getProperty<int>("HUNGER_TO_TRIGGER_CONSUMPTION")) {
            resources[0] -= 1;
            resources[1] -= 1;
            hunger -= FLAMEGPU->environment.getProperty<int>("HUNGER_PER_RESOURCE_CONSUMPTION");
        }
    }

    // reset analysis data
    FLAMEGPU->setVariable<int, N_DIM>("ana_last_resource_location", 0, -1);
    FLAMEGPU->setVariable<int, N_DIM>("ana_last_resource_location", 1, -1);

    // GOAP algorithm
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
    for (int resource_type = 0; resource_type < N_RESOURCE_TYPES; resource_type++) {
        float distance_to_resource =
            FLAMEGPU->getVariable<float, N_RESOURCE_TYPES>("closest_resource", resource_type);
        if (can_collect_resource &&
            distance_to_resource <=
                FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE")) {
            scores[Action::CollectResource0 + resource_type] =
                10 + (FLAMEGPU->environment.getProperty<float>("TARGET_RESOURCE_AMOUNT") -
                      resources[resource_type]);
        }
        if (can_move &&
            distance_to_resource >
                FLAMEGPU->environment.getProperty<float>("RESOURCE_COLLECTION_RANGE") &&
            distance_to_resource != FLT_MAX) {
            const float resource_saturation =
                FLAMEGPU->environment.getProperty<float>("TARGET_RESOURCE_AMOUNT") -
                resources[resource_type];
            scores[Action::MoveToClosestResource0 + resource_type] =
                int(10 - distance_to_resource * FLAMEGPU->environment.getProperty<float>(
                                                    "SCORE_REDUCTION_PER_TILE_DISTANCE")) +
                resource_saturation;
        }
    }
    int selected_action = findMax(scores, Action::EOF);
    // printAction(FLAMEGPU->getID(), selected_action);
    switch (selected_action) {
    case Action::RandomWalk:
        random_walk();
        break;
    case Action::Rest:
        rest();
        break;
    case Action::CollectResource0:
        collect_resource(0);
        break;
    case Action::CollectResource1:
        collect_resource(1);
        break;
    case Action::MoveToClosestResource0:
        move_to_closest_resource(0);
        break;
    case Action::MoveToClosestResource1:
        move_to_closest_resource(1);
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
    for (int resource_type = 0; resource_type < N_RESOURCE_TYPES; resource_type++) {
        FLAMEGPU->setVariable<int, N_RESOURCE_TYPES>("resources", resource_type,
                                                     resources[resource_type]);
    }
    return flamegpu::ALIVE;
}
