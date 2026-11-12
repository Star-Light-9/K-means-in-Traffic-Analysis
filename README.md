# Traffic clustering: sequential vs MPI-parallel DBSCAN (SUMO data)

SUMO simulates the traffic. Every vehicle is clustered at every step with
its position and velocity, once sequentially and once with MPI. Clusters
are tracked over time, flagged as flowing or stuck, and the two runs are
replayed side by side.

**Every setting lives in one file: `config.ini`.** Edit it, save, run the
next command.

## Files

| file | job |
|---|---|
| `config.ini` | all parameters (SUMO, DBSCAN, MPI, tracking, replay/GIF, benchmark) |
| `run_sumo.py` | runs SUMO, writes `trajectories.csv` |
| `dbscan.c` | sequential + MPI DBSCAN, writes clusters and timings |
| `run_dbscan.py` | optional: builds and starts the MPI run with the right command for your OS |
| `track.py` | follows clusters over time, flags stuck ones |
| `replay.py` | side-by-side animation, optional GIF |
| `benchmark.py` | speedup / efficiency / load balance for several process counts (results in `data/bench/`) |
| `config.py` | helper that reads `config.ini` |

## 1. One-time setup

```
pip install -r requirements.txt
```

Install MPI and a C compiler:

* **Linux / WSL:** `sudo apt install build-essential openmpi-bin libopenmpi-dev`
* **macOS:** `brew install open-mpi`
* **Windows:** WSL (Ubuntu) is the easiest, then use the Linux commands.
  Native Windows needs MS-MPI (runtime and SDK) plus MinGW gcc.

Build the clustering program once:

```
mpicc -O2 dbscan.c -o dbscan -lm                  # Linux / macOS / WSL
gcc -O2 dbscan.c -o dbscan.exe -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -lm    # Windows (MS-MPI)
```

(`python run_dbscan.py` does this for you automatically if it is missing.)

## 2. Run order

Run everything from this folder.

```
python run_sumo.py                 # 1. simulate traffic  -> data/trajectories.csv

mpirun -n 4 ./dbscan               # 2. cluster (Linux / macOS / WSL)
mpiexec -n 4 dbscan.exe            #    cluster (Windows, MS-MPI)
python run_dbscan.py               #    or: same thing, OS detected, -n taken from config.ini

python track.py                    # 3. track clusters, flag stuck ones
python replay.py                   # 4. side-by-side animation (+ GIF if enabled)
python benchmark.py                # 5. optional: speedup for 1, 2, 4, 8 processes
```

Step 2 note: the number after `-n` is the number of MPI processes. With
`run_dbscan.py` it comes from `[mpi] processes` in `config.ini`.

Outputs go to `data/` (or whatever `data_dir` says): `trajectories.csv`,
`clusters_seq.csv`, `clusters_par.csv`, `timings.csv`, `tracked_*.csv`,
`cluster_stats_*.csv`, `replay.gif`, and `bench/` for the benchmark.

## 3. Which step to rerun after changing a setting

| you changed | rerun from |
|---|---|
| `[sumo]` | step 1 (everything) |
| `[dbscan]` or `[mpi]` | step 2 |
| `[tracking]` | step 3 |
| `[replay]` | step 4 |
| `[benchmark]` | step 5 |

## 4. The settings in short

* `[sumo]`: `network` (`grid` or your own `.net.xml`), grid size, number of
  steps, warm-up, initial vehicles, new vehicles per step, number of jams,
  when they start and how long they last, random seed.
* `[dbscan]`: `eps_pos` (metres), `eps_vel` (m/s), `min_pts`, `split`
  (`balanced` or `equal`), `reps`, `max_steps`.
  Two vehicles are neighbours when closer than `eps_pos` **and** their
  velocity vectors differ by less than `eps_vel`. Velocity is compared as a
  vector `(vx, vy)`, so opposite directions never share a cluster.
* `[mpi]`: `processes`, `launcher`, `extra_args` (for example
  `--oversubscribe` with OpenMPI when processes > cores).
* `[tracking]`: `stuck_speed`, `stuck_steps`. A cluster is stuck when the
  mean speed of its vehicles stays under `stuck_speed` for `stuck_steps`
  steps in a row.
* `[replay]`: `seq_duration`, `fps`, `show_window`, `make_gif`, `gif_name`.
* `[benchmark]`: `processes`, `reps`, `compare_split`.

You can also override a clustering setting for one run without editing the
file, for example `mpirun -n 4 ./dbscan config.ini reps=3 split=equal`.

## 5. The replay

Both panels start at step 0 together. Each advances only after the real
compute time it needed (from `timings.csv`) has elapsed, so the faster
version visibly gets ahead. Real times are milliseconds, so they are
stretched by one common factor so the sequential run lasts `seq_duration`
seconds on screen. Cars are coloured by cluster, grey is noise, a black
outline marks a stuck cluster.

If `show_window = true` but your Python has no GUI toolkit, `replay.py`
writes the GIF instead. For a live window on Linux install
`python3-tk`, or try `pip install PyQt6`.

## 6. How the parallel clustering works (per time step)

1. Every rank holds the whole step (the vehicle feed is broadcast once).
2. The map is cut into strips along its longer side. With `split = balanced`
   every rank owns the same NUMBER of vehicles; with `equal` the same width.
3. Each rank clusters its strip plus a halo of width `eps_pos`.
4. Rank 0 gathers the labels and merges clusters that touch across strip
   borders with union-find.

The timed parallel region includes partitioning, local clustering, the
gather and the merge, not file output.

## 7. Getting trustworthy timings

* Close other programs (browser, video, downloads) during step 2 and 5.
  Plug in a laptop and use a high-performance power mode.
* Use no more processes than physical CPU cores.
* Raise `reps` (for example 3 to 5) for steadier numbers.
* **Small steps will not speed up.** A step with 1-3 thousand cars clusters
  in about a millisecond, so MPI overhead can make the parallel version
  slower. Speedup appears with tens of thousands of cars per step. Raise
  `initial_vehicles` and `flow` in `[sumo]` (and use a bigger `grid_size`)
  to test that, and report both cases: the crossover is a result.
* Parallel and sequential results can differ slightly (for example a border
  car that could belong to two clusters). They are not forced to match.
* The sequential pass runs on rank 0 while the other ranks wait in a
  barrier. Some MPI builds busy-wait there, which can slow the sequential
  timing a little if processes > cores.

## 8. Troubleshooting

* **`data/data/trajectories.csv` not found, or `bench/p1_bal1/timings.csv`
  not found:** an old script is being used with the new `dbscan.c`. All
  files must come from the same version. Replace every file with the ones
  from this package (especially `benchmark.py`), recompile `dbscan.c`, and
  do not hand-edit paths: all paths come from `data_dir` in `config.ini`.
* **`Cannot open data/trajectories.csv`:** run `python run_sumo.py` first,
  and run the other commands from the folder that contains `config.ini`
  (or pass its path: `./dbscan path/to/config.ini`, `--config path` for the
  Python scripts).
* **`dbscan not found` (benchmark):** compile `dbscan.c` into this folder.
* **`mpirun` / `mpiexec` not found:** install MPI (section 1). On Windows set
  `launcher = mpiexec` in `[mpi]`.
* **More processes than cores (OpenMPI):** add `extra_args = --oversubscribe`
  in `[mpi]`, or lower `processes`.
* **No window opens in `replay.py`:** your Python has no GUI toolkit. Keep
  `make_gif = true` and open the GIF, or install `python3-tk` /
  `pip install PyQt6`.

## 9. Using your own road network (for example Pune)

Export an OpenStreetMap extract, convert it with SUMO's `netconvert`, put
the `.net.xml` in this folder and set `network = pune.net.xml` in
`config.ini`:

```
netconvert --osm-files pune.osm -o pune.net.xml --geometry.remove --roundabouts.guess --junctions.join --tls.guess-signals --tls.discard-simple
```

SUMO real-world traffic is usually sparser than a dense grid, so you may
need a larger `eps_pos` (30-40) and `min_pts` 3.