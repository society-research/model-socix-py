import random

import ostruct
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
    sinkhorn = ot.sinkhorn(a, b, M_loss_jiggled, regularization, numItermax=10000)
    # plot_ot(pos_source_j * SCALE, pos_target_j * SCALE, sinkhorn, "sinkhorn")

    # EMD -- Earth Movers Distance - works fine
    a, b = np.ones((n,)) / n, np.ones((m,)) / m  # uniform distribution on samples
    emd = ot.emd(a, b, M_loss)
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
    alive_humans = []
    avg_resources = []
    for step in range(steps):
        simulation.step()
        simulation.getPopulationData(humans)
        alive_humans.append(len(humans))
        if len(humans) == 0:
            print("[WARNING] All humans are dead. Simulation stops early.")
            break
        avg_resources.append(np.array([0, 0], dtype="float64"))
        for human in humans:
            id = human.getID()
            res = np.array(human.getVariableArrayInt("resources"))
            avg_resources[-1] += res
            x, y = human.getVariableInt("x"), human.getVariableInt("y")
            paths.append([id, x, y])
            loc = human.getVariableArrayInt("ana_last_resource_location")
            if loc != (-1, -1):
                collected_resources.append([id, *loc])
        avg_resources[-1] /= len(humans)
    return (
        util.collected_resource_list_to_cost_matrix(
            collected_resources, pos_source, pos_target
        ),
        {
            "paths": paths,
            "alive_humans": alive_humans,
            "avg_resources": avg_resources,
            "constants": C,
        },
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


def plot_ot(meta, xs, xt, solution, what, extra=""):
    plt.figure(figsize=(18, 6))  # Adjust the figure size as needed
    # Plot 1: Cost matrix M
    plt.subplot(1, 3, 1)
    plt.imshow(meta["loss"], interpolation="nearest")
    plt.title(f"Cost matrix M{extra}")
    # Plot 2: OT matrix `solution`
    plt.subplot(1, 3, 2)
    plt.imshow(solution, interpolation="nearest")
    plt.title(f"OT matrix: {what}")
    # Plot 3: OT matrix `solution` with samples
    plt.subplot(1, 3, 3)
    ot.plot.plot2D_samples_mat(xs, xt, solution, color=[0.5, 0.5, 1])
    plt.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
    plt.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
    plt.legend(loc=0)
    plt.title(f"OT matrix: {what} with samples")
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()


# TODO: should penalize every single row (and column) that does not sum up to
#       1/(len(rows) | 1/(len(columns) respectively, since this means it is not
#       saturated, i.e. mines/factories are not working 100%
def loss(M_cost, M_solution):
    """loss is defined as the cost of transport from xs[i] to xt[i] (i.e.
    M_cost[i,j] multiplied with the amount (M_sol[i,j])
    """
    l = np.sum(M_cost * M_solution.T)
    # penalty for avoiding a source/target spot
    avg_dist = np.mean(M_cost)
    missing_sources = np.sum(np.all(M_solution == 0, axis=1))  # rows
    missing_targets = np.sum(np.all(M_solution == 0, axis=1))  # columns
    missing = missing_sources + missing_targets
    if missing > 0:
        l += missing * avg_dist
    return l


def compare(xs, xt, solution_abm, solution_ot):
    """compare returns a comparison object with various metrics comparing the
    input solutions.

    Returns
    -------
        comparison (OpenStruct)
            .loss_ot        loss of the OT solution
            .loss_abm       loss of the ABM solution
    """
    comparison = ostruct.OpenStruct()
    M_loss = ot.dist(xs, xt)
    comparison.loss_ot = loss(M_loss, solution_ot)
    comparison.loss_abm = loss(M_loss, solution_abm)
    return comparison


def plot_abm(meta, xs, xt, solution, config, comparison):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Create a 2x3 grid of subplots
    # Plot 1: ABM metrics
    humans = np.array([i for i in enumerate(meta["alive_humans"])])
    x, y = humans[:, 0], humans[:, 1]
    ax = axs[0, 0]
    ax.plot(x, y, "b-", label="Humans Alive")
    ax.set_ylabel("no. humans alive / 1")
    ax.set_ylim(top=np.max(y) * 1.05)
    ax.set_xlabel("step / 1")
    ax.set_title("Alive humans")
    ax.legend(loc="upper left")
    # Plot 2: OT matrix `solution`
    ax = axs[0, 1]
    ax.imshow(solution, interpolation="nearest")
    ax.set_title(f"OT matrix: ABM")
    # Plot 3: OT matrix `solution` with samples
    ax = axs[0, 2]
    plt.subplot(2, 3, 3)  # 2x3 plots, this is no 3 of 6 (=2x3)
    ot.plot.plot2D_samples_mat(xs, xt, solution, color=[0.5, 0.5, 1])
    ax.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
    ax.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
    ax.legend(loc=0)
    ax.set_title(f"OT matrix: ABM with samples")
    # Additional plot in the second row, first column
    ax = axs[1, 0]
    res = np.array(
        [[step, r1, r2] for step, [r1, r2] in enumerate(meta["avg_resources"])]
    )
    ax.plot(res[:, 0], res[:, 1], "r-", label="average resource[type=0]")
    ax.plot(res[:, 0], res[:, 2], "b-", label="average resource[type=1]")
    ax.set_ylabel("Average resources per human")
    ax.set_ylim(top=np.max([res[:, 1], res[:, 2]]) * 1.05)
    ax.set_xlabel("step / 1")
    ax.set_title("Average Resources per human (averaged over ticks)")
    ax.legend(loc="upper left")
    # Text field in the second row, second column
    ax = axs[1, 1]
    config = ostruct.OpenStruct(**config)
    text = f"""ABM configuration:
    n_humans                        {config.n_humans}
    seed                            {config.seed}
    resource_restoration_ticks      {config.resource_restoration_ticks}
    hunger_starved_to_death         {config.hunger_starved_to_death}
    n_humans_crowded                {config.n_humans_crowded}
    steps                           {config.steps}

    loss_ot                         {comparison.loss_ot}
    loss_abm                        {comparison.loss_abm}
    """
    ax.text(
        0.5,
        0.5,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14,
    )
    ax.axis("off")  # Hide the axis
    # Hide the unused second row, third column
    ax = axs[1, 2]
    ax.axis("off")
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
    min, max = 1, 1
    xs = np.round(xs).astype("int")
    xt = np.round(xt).astype("int")
    return _make_distrib_unique(xs), _make_distrib_unique(xt)
    # return xs, xt


def _make_distrib_unique(x):
    orig_len = len(x)
    x = np.unique(x, axis=0)
    min = np.array([np.min(x[:, 0]), np.min(x[:, 1])])
    max = np.array([np.max(x[:, 0]), np.max(x[:, 1])])
    if (min >= max).any():
        max += 1
    while orig_len > len(x):
        x = np.append(
            x,
            [[np.random.randint(min[0], max[0]), np.random.randint(min[1], max[1])]],
            axis=0,
        )
        x = np.unique(x, axis=0)
    return x


class Optimizer:
    def __init__(self, **config):
        self.config = ostruct.OpenStruct(**config)

    def all(self):
        for name, param in self.config.items():
            for val in param.all():
                yield ostruct.OpenStruct({name: val})


class HyperParameter:
    def __init__(self, min=None, max=None, steps=None):
        """Initialize with range [min, max) testing with step size steps."""
        self.min = min
        self.max = max
        self.steps = steps

    def all(self):
        return range(self.min, self.max, self.steps)
