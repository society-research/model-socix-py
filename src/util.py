import numpy as np


def collected_resource_list_to_cost_matrix(
    collections, srcLocations, tgtLocations, use_last_only=False
):
    cost = np.zeros((len(srcLocations), len(tgtLocations)), dtype="float")
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
        x, y = x[0], y[0]
        return x, y

    def is_valid(path):
        x, y = get_resource_slot(path)
        if x is not None and y is not None:
            return True
        return False

    for id, events in agents.items():
        it = iter(events)
        prefix = None
        while True:
            src, tgt = [], []
            if prefix is not None:
                src.append(prefix)
                prefix = None
            try:
                if len(src) == 0:
                    src.append(next(it))
                tgt.append(next(it))
                while (src[0] == tgt[-1]).all():
                    src.append(tgt.pop())
                    tgt.append(next(it))
                while not is_valid([src[0], tgt[-1]]):
                    src.append(tgt.pop())
                    tgt.append(next(it))
                stop = False
                while is_valid([src[0], tgt[-1]]):
                    try:
                        tgt.append(next(it))
                    except StopIteration:
                        stop = True
                        break
                if not stop:
                    prefix = tgt.pop()
                if use_last_only:
                    src = [src[-1]]
                    tgt = [tgt[0]]
                for i in range(len(src)):
                    for j in range(len(tgt)):
                        x, y = get_resource_slot([src[i], tgt[j]])
                        cost[x, y] += 1 / (len(src) * len(tgt))
                if stop:
                    break
            except StopIteration:
                break
    if np.sum(cost) == 0:
        return cost
    return cost / np.sum(cost)
