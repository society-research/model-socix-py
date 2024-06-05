import math

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot
import optimal_transport as solver

try:
    Path("./output/").mkdir(parents=True)
except FileExistsError:
    pass

xs, xt = solver.generate_distributions()
SCALE = 5
sinkhorn, sinkhorn_meta = solver.solve_ot_with_sinkhorn(xs, xt, SCALE=SCALE)
solver.plot_ot(
    sinkhorn_meta, xs, xt, sinkhorn, "sinkhorn", file="output/sinkhorn-solution.svg"
)
