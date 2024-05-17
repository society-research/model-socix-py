import random

import numpy as np
from matplotlib import pyplot as plt
import ot
import util
import pyflamegpu

from sx import make_simulation, C


def solve_ot_with_sinkhorn(pos_source, pos_target, SCALE=5, jiggle_factor=0.01):
    """solve_ot_with_sinkhorn uses the sinkhorn algorithm to solve the optimal
        transport problem between distributions pos_source and pos_target.

    Arguments:
        SCALE (int): see generate_distributions
        jiggle_factor (float): amount of jiggling applied to resource locations
    """
    M_loss = ot.dist(pos_source, pos_target)
    jiggle = lambda x: (x + jiggle_factor * np.random.rand(*x.shape)) / SCALE
    pos_source_j, pos_target_j = jiggle(pos_source), jiggle(pos_target)

    # NOTE: To apply sinkhorns algorithm the distributions must not have
    # resources at the exact same distances, thus locations are "jiggled" a bit.
    M_loss_jiggled = ot.dist(pos_source_j, pos_target_j)

    # plot_cost(M_loss)
    # plot_cost(M_loss_jiggled, extra="_jiggled")

    # initialize
    n, m = len(pos_source), len(pos_target)

    # sinkhorn: needs x,y \in [0, 1] and requires jiggled input for convergence
    # (equal distances that are common with integer locations on a small scale hinder convergence)
    a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform distribution on samples
    regularization = 1e-1
    sinkhorn = ot.sinkhorn(a, b, M_loss_jiggled, regularization)
    # print(f"sinkhorn:\n{sinkhorn}")
    # plot_ot(pos_source_j * SCALE, pos_target_j * SCALE, sinkhorn, "sinkhorn")

    # EMD -- Earth Movers Distance - works fine
    a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform distribution on samples
    emd = ot.emd(a, b, M_loss)
    # print(f"EMD:\n{emd}")
    # plot_ot(pos_source, pos_target, emd, "EMD")

    return sinkhorn, {"emd": emd, "loss": M_loss, "loss_jiggled": M_loss_jiggled}


def solve_ot_with_abm(
    pos_source,
    pos_target,
    seed=4,
    n_humans=100,
    steps=1000,
    resource_restoration_ticks=50,
    hunger_starved_to_death=1000,
    n_humans_crowded=200,
):
    """Solver for the optimal transport problem using the ABM sx.py"""
    random.seed(seed)
    grid_size = int(np.max([pos_source, pos_target]))
    C.RESOURCE_RESTORATION_TICKS = resource_restoration_ticks
    C.HUNGER_STARVED_TO_DEATH = hunger_starved_to_death
    C.N_HUMANS_CROWDED = n_humans_crowded
    model, simulation, ctx = make_simulation(grid_size=grid_size)
    simulation.SimulationConfig().random_seed = seed
    resources = pyflamegpu.AgentVector(ctx.resource, len(pos_source) + len(pos_target))
    for i, p in enumerate(pos_source):
        resources[i].setVariableInt("x", int(p[0]))
        resources[i].setVariableInt("y", int(p[1]))
        resources[i].setVariableInt("type", 0)
    for i, p in enumerate(pos_target):
        resources[i + len(pos_source)].setVariableInt("x", int(p[0]))
        resources[i + len(pos_source)].setVariableInt("y", int(p[1]))
        resources[i + len(pos_source)].setVariableInt("type", 1)
    humans = pyflamegpu.AgentVector(ctx.human, n_humans)
    for human in humans:
        human.setVariableInt("x", random.randint(0, grid_size))
        human.setVariableInt("y", random.randint(0, grid_size))
        human.setVariableArrayInt("resources", (2, 2))
        human.setVariableFloat("actionpotential", C.AP_DEFAULT)
    for av in [resources, humans]:
        simulation.setPopulationData(av)
    paths = []
    collected_resources = []
    for step in range(steps):
        simulation.step()
        simulation.getPopulationData(humans)
        if len(humans) == 0:
            print("[WARNING] All humans are dead. Simulation stops early.")
            break
        if step % (steps / 10) == (steps / 10) - 1:
            print(f"[{step}] humans={len(humans)}")
        for human in humans:
            id = human.getID()
            x, y = human.getVariableInt("x"), human.getVariableInt("y")
            paths.append([id, x, y])
            loc = human.getVariableArrayInt("ana_last_resource_location")
            if loc != (-1, -1):
                collected_resources.append([id, *loc])
    return (
        util.collected_resource_list_to_cost_matrix(
            collected_resources, pos_source, pos_target
        ),
        paths,
    )


def plot_paths_4x4(pos_source, pos_target, paths):
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    axs = axs.flatten()
    paths = np.array(paths)
    grid_size = int(np.max([pos_source, pos_target]))
    for i, id in enumerate(set(paths[:, 0])):
        if i == 16:
            break  # only size for 16 humans in this grid
        path = paths[paths[:, 0] == id][:, 1:3]
        x = path[:, 0]
        y = path[:, 1]
        ax = axs[i]
        ax.scatter(
            pos_source[:, 0],
            pos_source[:, 1],
            color="green",
            marker="x",
            label="resource[0]",
        )
        ax.scatter(
            pos_target[:, 0],
            pos_target[:, 1],
            color="orange",
            marker="x",
            label="resource[1]",
        )
        ax.plot(x, y, linestyle="-", marker="", label=f"path of H[{id}]")
        ax.set_title(f"Human[{id}]")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_cost(M, extra=""):
    plt.figure(2)
    plt.imshow(M, interpolation="nearest")
    plt.title(f"Cost matrix M{extra}")
    plt.show()


def plot_ot(xs, xt, G, what):
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    plt.subplot(1, 2, 1)
    plt.imshow(G, interpolation="nearest")
    plt.title(f"OT matrix: {what}")
    plt.subplot(1, 2, 2)
    ot.plot.plot2D_samples_mat(xs, xt, G, color=[0.5, 0.5, 1])
    plt.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
    plt.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
    plt.legend(loc=0)
    plt.title(f"OT matrix: {what} with samples")
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()


def generate_distributions(s=50, t=50, SCALE=5):
    """generate_distributions generates a distribution of 2 sample sets in the
    coordinate range [0, SCALE].

    Arguments:
        SCALE (int): scaling x,y from [0, 1] to a range that can be reasonably
            displayed in integer coordinates, and that is not too large for
            agents to walk through.
    """
    mu_s = np.array([4, 4])
    cov_s = np.array([[1, 0], [0, 1]])
    mu_t = np.array([8, 8])
    cov_t = np.array([[1, -0.8], [-0.8, 1]])
    xs = ot.datasets.make_2D_samples_gauss(s, mu_s, cov_s)
    xt = ot.datasets.make_2D_samples_gauss(t, mu_t, cov_t)
    xs[xs < 0] *= -1
    xt[xt < 0] *= -1
    xs *= SCALE
    xt *= SCALE
    return np.round(xs).astype("int"), np.round(xt).astype("int")
