import numpy as np


def collected_resource_list_to_cost_matrix(collections, srcLocations, tgtLocations):
    cost = np.zeros((len(srcLocations), len(tgtLocations)))
    agents = {}
    for event in collections:
        event = np.array(event)
        if event[0] not in agents.keys():
            agents[event[0]] = []
        agents[event[0]].append(event[1:])

    def get_resource_slot(path):
        x = np.where((srcLocations == path[0]).all(axis=1))[0]
        y = np.where((tgtLocations == path[1]).all(axis=1))[0]
        if len(x) == 0 or len(y) == 0:
            x = np.where((srcLocations == path[1]).all(axis=1))[0]
            y = np.where((tgtLocations == path[0]).all(axis=1))[0]
        if len(x) == 0 or len(y) == 0:
            return None, None
            # raise RuntimeError(
            #    f"path={path} not found in srcLocations={srcLocations} or tgtLocations={tgtLocations}"
            # )
        return x[0], y[0]

    for id, events in agents.items():
        it = iter(events)
        while True:
            try:
                first = next(it)
                second = next(it)
                if (first == second).all():
                    first = second
                    second = next(it)
                if (first == second).all():
                    continue
                x, y = get_resource_slot([first, second])
                # TODO: extend to arbitrary path lengths
                if x is None or y is None:
                    third = next(it)
                    x, y = get_resource_slot([first, third])
                    cost[x, y] += 0.2
                    x, y = get_resource_slot([second, third])
                    cost[x, y] += 0.8
                else:
                    cost[x, y] += 1
            except StopIteration:
                break
    if np.sum(cost) == 0:
        return cost
    return cost / np.sum(cost)
