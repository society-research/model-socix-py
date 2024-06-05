import math
import os

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot
import optimal_transport as solver

# font handling for thesis
from matplotlib import font_manager, rcParams

## Set font properties for matplotlib
# rcParams['font.family'] = 'DejaVu Serif'
# rcParams['font.sans-serif'] = ['DejaVu Sans']
# rcParams['font.monospace'] = ['DejaVu Sans Mono']
## libertinus does not work
rcParams["font.family"] = "Libertinus Serif"
rcParams["font.sans-serif"] = ["Libertinus Sans"]
rcParams["font.monospace"] = ["Libertinus Mono"]
# Path to your local font directory
local_font_path = os.getenv("HOME") + "/.local/share/fonts/Libertinus"
# Add your local font directory to the font manager
font_dirs = [local_font_path]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs, fontext="otf")
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


try:
    Path("./output/").mkdir(parents=True)
except FileExistsError:
    pass


def plot_ot(meta, xs, xt, solution, file=""):
    # size
    sz = 8
    # rows and columns
    r, c = 2, 2
    plt.figure(figsize=(sz, sz))  # Adjust the figure size as needed
    plt.subplot(r, c, 1)
    plt.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
    plt.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
    plt.xlabel("x / 1")
    plt.ylabel("y / 1")
    plt.legend(loc=0)
    plt.title(f"Source and target distributions")
    # Plot 0: Distributions
    # Plot 1: Cost matrix M
    plt.subplot(r, c, 2)
    plt.imshow(meta["loss"], interpolation="nearest")
    plt.xlabel("Index of source / 1")
    plt.ylabel("Index of target / 1")
    plt.title(f"Cost matrix M")
    # Plot 2: OT matrix `solution`
    plt.subplot(r, c, 3)
    plt.imshow(solution, interpolation="nearest")
    plt.xlabel("Index of source / 1")
    plt.ylabel("Index of target / 1")
    plt.title(f"Transport plan: sinkhorn")
    # Plot 3: OT matrix `solution` with samples
    plt.subplot(r, c, 4)
    ot.plot.plot2D_samples_mat(xs, xt, solution, color=[0.5, 0.5, 1])
    plt.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
    plt.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
    plt.xlabel("x / 1")
    plt.ylabel("y / 1")
    plt.legend(loc=0)
    plt.title(f"Transport plan: sinkhorn with samples")
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.savefig(file)
    plt.clf()


xs, xt = solver.generate_distributions()
SCALE = 5
sinkhorn, sinkhorn_meta = solver.solve_ot_with_sinkhorn(xs, xt, SCALE=SCALE)
plot_ot(sinkhorn_meta, xs, xt, sinkhorn, file="output/sinkhorn-solution.svg")
