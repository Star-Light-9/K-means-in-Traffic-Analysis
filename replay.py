"""
replay.py - side-by-side replay of the sequential and the parallel run.

Both panels start at step 0 at the same moment. Each panel moves on to
its next step only after the REAL compute time that version needed for
the steps so far has elapsed (taken from timings.csv). The faster
version therefore visibly gets ahead.

Because DBSCAN on one step takes only milliseconds, the real times are
stretched by a common factor so that the sequential run lasts
--seq-duration seconds on screen. Both panels use the same factor.

Cars are coloured by cluster, noise is grey, and clusters flagged as
stuck get a black outline.

Settings (duration, fps, window / GIF) come from the [replay] section of
config.ini.

Usage:
    python replay.py
    python replay.py --png still.png --at 0.4     (one still image)
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib

import config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None,
                help="settings file (default: config.ini next to the scripts)")
ap.add_argument("--png", default=None,
                help="write one still image and exit")
ap.add_argument("--at", type=float, default=0.5,
                help="with --png: moment as a fraction of the sequential run")
cli = ap.parse_args()

cfg = config.load(cli.config)

args = SimpleNamespace(
    data_dir=cfg.data_dir,
    seq_duration=cfg.float("replay", "seq_duration"),
    fps=cfg.int("replay", "fps"),
    show_window=cfg.bool("replay", "show_window"),
    make_gif=cfg.bool("replay", "make_gif"),
    gif_name=cfg.str("replay", "gif_name"),
    png=cli.png,
    at=cli.at,
)

if args.png or (args.make_gif and not args.show_window):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

D = args.data_dir

traj = pd.read_csv(os.path.join(D, "trajectories.csv"),
                   usecols=["step", "id", "x", "y"])
timings = pd.read_csv(os.path.join(D, "timings.csv"))

steps = timings["step"].to_numpy()
NF = len(steps)

# ---------------- per-mode data, indexed by step ----------------
cmap = plt.get_cmap("tab20")
palette = np.array([cmap(i % 20) for i in range(20)])
NOISE_COLOR = (0.62, 0.62, 0.62, 0.55)


def load_mode(mode):
    tr = pd.read_csv(os.path.join(D, "tracked_%s.csv" % mode))
    m = traj.merge(tr, on=["step", "id"], how="left")
    m["track"] = m["track"].fillna(-1).astype(int)
    m["stuck"] = m["stuck"].fillna(0).astype(int)

    frames = {}

    for step, g in m.groupby("step"):
        track = g["track"].to_numpy()
        stuck = g["stuck"].to_numpy()

        colors = np.empty((len(g), 4))
        colors[:] = NOISE_COLOR
        ok = track >= 0
        colors[ok] = palette[track[ok] % 20]

        n_clusters = len(np.unique(track[ok]))
        n_stuck = len(np.unique(track[(stuck == 1) & ok]))

        frames[step] = {
            "xy": g[["x", "y"]].to_numpy(),
            "colors": colors,
            "stuck": stuck.astype(bool),
            "clusters": n_clusters,
            "stuck_clusters": n_stuck,
            "noise": int((~ok).sum()),
        }

    return frames


seq_frames = load_mode("seq")
par_frames = load_mode("par")

# ---------------- timing -> screen time ----------------
seq_real = np.cumsum(timings["seq_ms"].to_numpy()) / 1000.0
par_real = np.cumsum(timings["par_ms"].to_numpy()) / 1000.0

scale = args.seq_duration / seq_real[-1]
seq_ready = seq_real * scale
par_ready = par_real * scale

total_time = max(seq_ready[-1], par_ready[-1]) + 3.0
dt = 1.0 / args.fps
n_anim = int(total_time / dt)


def frame_index(ready, t):
    """Index of the latest step whose computation has finished at time t."""
    k = int(np.searchsorted(ready, t, side="right")) - 1
    return max(k, 0), k >= NF - 1 and t >= ready[-1]


# ---------------- figure ----------------
all_xy = traj[["x", "y"]].to_numpy()
pad = 0.03 * max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]))
xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
fig.patch.set_facecolor("#fafafa")

net_path = os.path.join(D, "network.json")
net_lines = []

if os.path.isfile(net_path):
    with open(net_path) as f:
        net_lines = json.load(f)

panels = []

for ax, name, frames, ready in (
    (axes[0], "Sequential DBSCAN", seq_frames, seq_ready),
    (axes[1], "Parallel DBSCAN (MPI)", par_frames, par_ready),
):
    ax.set_facecolor("#ffffff")

    for line in net_lines:
        arr = np.array(line)
        ax.plot(arr[:, 0], arr[:, 1], color="#dddddd", lw=0.8, zorder=1)

    first = frames[steps[0]]
    sc = ax.scatter(first["xy"][:, 0], first["xy"][:, 1],
                    s=14, c=first["colors"], edgecolors="none", zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    title = ax.set_title(name, fontsize=13, fontweight="bold", loc="left")
    info = ax.text(0.01, 0.01, "", transform=ax.transAxes, fontsize=10,
                   va="bottom", ha="left", zorder=5,
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    panels.append((ax, sc, info, title, frames, ready,
                   timings["seq_ms" if "Sequential" in name else "par_ms"]
                   .to_numpy().cumsum() / 1000.0))

clock = fig.text(0.5, 0.965, "", ha="center", fontsize=11)
fig.text(0.5, 0.01,
         "Both runs start at step %d together. A panel only advances when "
         "its real compute time has elapsed (time axis stretched x%.0f, "
         "same for both).  Grey = noise, black outline = stuck cluster."
         % (steps[0], scale),
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))


def draw(k):
    t = k * dt
    clock.set_text("screen time %.1f s" % t)

    for ax, sc, info, title, frames, ready, real_cum in panels:
        i, done = frame_index(ready, t)
        fr = frames[steps[i]]

        sc.set_offsets(fr["xy"])
        sc.set_facecolor(fr["colors"])

        edge = np.zeros((len(fr["xy"]), 4))
        edge[fr["stuck"]] = (0.0, 0.0, 0.0, 1.0)
        sc.set_edgecolor(edge)
        sc.set_linewidth(np.where(fr["stuck"], 1.4, 0.0))

        info.set_text(
            "step %d / %d%s\n"
            "compute time so far: %.3f s\n"
            "clusters: %d   stuck: %d   noise: %d"
            % (steps[i], steps[-1], "   FINISHED" if done else "",
               real_cum[i], fr["clusters"], fr["stuck_clusters"], fr["noise"]))

    return []


if args.png:
    draw(int(args.at * args.seq_duration / dt))
    fig.savefig(args.png, dpi=130)
    print("Wrote", args.png)
    sys.exit(0)

anim = FuncAnimation(fig, draw, frames=n_anim, interval=1000 / args.fps,
                     blit=False, repeat=False)

no_window = {"agg", "pdf", "pgf", "ps", "svg", "cairo", "template"}
window_possible = matplotlib.get_backend().lower() not in no_window
want_gif = args.make_gif

if args.show_window and not window_possible:
    print("No window available (matplotlib backend: %s)."
          % matplotlib.get_backend())
    print("To get a live window install a GUI toolkit: on Linux "
          "'sudo apt install python3-tk', or try 'pip install PyQt6'.")

    if not want_gif:
        print("Writing the GIF instead.")
        want_gif = True

if want_gif:
    gif_path = os.path.join(D, args.gif_name)
    print("Writing %s (this can take a few minutes; lower seq_duration or "
          "fps in config.ini to speed it up) ..." % gif_path)

    if gif_path.lower().endswith(".mp4"):
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=args.fps)
    else:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=args.fps)

    anim.save(gif_path, writer=writer, dpi=90)
    print("Wrote", gif_path)

if args.show_window and window_possible:
    plt.show()