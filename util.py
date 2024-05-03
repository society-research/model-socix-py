import numpy as np


def collected_resource_list_to_cost_matrix(collections, srcLocations, tgtLocations):
    cost = np.zeros((len(srcLocations), len(tgtLocations)))
    agents = {}
    for event in collections:
        event = np.array(event)
        if event[0] not in agents.keys():
            agents[event[0]] = []
        agents[event[0]].append(event[1:])
    for id, events in agents.items():
        for i in range(0, len(events) - 1):
            srcLocation, tgtLocation = events[i], events[i + 1]
            if (srcLocation == tgtLocation).all():
                continue
            x = np.where((srcLocations == srcLocation).all(axis=1))[0]
            y = np.where((tgtLocations == tgtLocation).all(axis=1))[0]
            if len(x) == 0:
                raise RuntimeError(
                    f"src={srcLocation} not found in locations={srcLocations}"
                )
            if len(y) == 0:
                raise RuntimeError(
                    f"tgt={tgtLocation} not found in locations={tgtLocations}"
                )
            x = x[0]
            y = y[0]
            cost[x, y] += 1
    return cost
