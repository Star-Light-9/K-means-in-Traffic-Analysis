# Traffic clustering: sequential vs MPI-parallel DBSCAN

Cluster vehicles at every simulation step using position and velocity,
once sequentially and once with MPI, then replay both side by side.

```
run_sumo.py  (or make_synthetic.py)  ->  data/trajectories.csv
dbscan.c  (sequential + MPI)         ->  data/clusters_seq.csv, clusters_par.csv, timings.csv
track.py                             ->  data/tracked_seq.csv, tracked_par.csv
replay.py                            ->  two-panel animation
benchmark.py                         ->  speedup / efficiency / load-imbalance charts
```

## 1. Install

```
pip install numpy pandas matplotlib
pip install eclipse-sumo traci sumolib      # only for run_sumo.py
```

MPI: OpenMPI or MPICH on Linux/macOS/WSL (`mpicc`, `mpirun`), or
MS-MPI on Windows.

## 2. Get trajectories

**With SUMO** (built-in grid city, jams created by temporary blockages):

```
python run_sumo.py --out-dir data
```

Useful options: `--grid 8` (city size), `--flow 3` (new cars per step),
`--initial 800`, `--jams 3`, `--jam-duration 120`, `--steps 400`.

**Without SUMO** (synthetic roads with platoons and bottlenecks, handy for
trying the pipeline or for very large scaling tests):

```
python make_synthetic.py --out-dir data
python make_synthetic.py --out-dir data_big --vehicles 40000 --steps 60
```

Both write `trajectories.csv` with columns `step,id,x,y,vx,vy,speed`
(metres, m/s) and `network.json` (road lines for the background).

## 3. Build and run the clustering

Linux / macOS / WSL:

```
mpicc -O2 dbscan.c -o dbscan -lm
mpirun -n 4 ./dbscan data/trajectories.csv data
```

Windows with MS-MPI (typical command, adjust to your setup):

```
gcc -O2 dbscan.c -o dbscan.exe -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -lm
mpiexec -n 4 dbscan.exe data\trajectories.csv data
```

The output folder must already exist. Full argument list:

```
dbscan <trajectories.csv> <outdir> [eps_pos=25] [eps_vel=4] [minpts=4]
       [balance=1] [reps=1] [max_frames=0]
```

| argument | meaning |
|---|---|
| `eps_pos` | two cars are neighbours if closer than this many metres ... |
| `eps_vel` | ... and their velocity vectors differ by less than this (m/s) |
| `minpts` | neighbours (including itself) needed for a core point |
| `balance` | 1 = strips with equal vehicle COUNT, 0 = equal WIDTH |
| `reps` | repeat each step this many times and average the timing |
| `max_frames` | only process the first N steps (0 = all) |

Velocity is compared as a vector `(vx, vy)`, so cars driving in opposite
directions never share a cluster, and there is no 359 deg / 1 deg angle
wrap-around problem.

### How the parallel version works (per time step)
1. Every rank has the whole step (the vehicle feed is broadcast once).
2. The map is cut into strips along its longer side. Strip borders are
   chosen so each rank owns the same NUMBER of vehicles (or the same width
   with `balance=0`).
3. Each rank clusters its strip plus a halo of width `eps_pos`.
4. Rank 0 gathers the labels and merges local clusters that touch across
   strip borders with union-find.

The timed region for the parallel version includes partitioning, local
clustering, the gather and the merge, not file output.

## 4. Track clusters and flag stuck ones

```
python track.py --data-dir data
```

A cluster keeps its id (and colour) from step to step by matching shared
vehicles. Its speed is the mean speed of its members. It is flagged
**stuck** when that stays below `--stuck-speed` (default 2 m/s) for
`--stuck-steps` (default 3) steps in a row.

## 5. Side-by-side replay

```
python replay.py --data-dir data
python replay.py --data-dir data --seq-duration 45
python replay.py --data-dir data --save replay.gif
python replay.py --data-dir data --png still.png --at 0.4
```

Both panels start at step 0 together. Each advances only after the real
compute time it needed (from `timings.csv`) has elapsed, so the faster
version gets ahead. Real times are a few milliseconds, so they are
stretched by one common factor so the sequential run lasts
`--seq-duration` seconds. Cars are coloured by cluster, grey = noise,
black outline = stuck cluster.

## 6. Benchmark

```
python benchmark.py --input data_big/trajectories.csv --procs 1 2 4 8 --reps 3 --compare-split
```

Windows: add `--mpirun mpiexec --exe dbscan.exe`.
OpenMPI with more processes than cores: `--mpi-extra --oversubscribe`.

Writes `bench/benchmark_results.csv` and `bench/benchmark.png`
(total time, speedup, load imbalance; count-balanced vs equal-width).

## Using a real Pune road network (untested here)

Get an OpenStreetMap extract of the area (for example with
`python %SUMO_HOME%/tools/osmWebWizard.py` or by exporting a .osm file),
then convert it:

```
netconvert --osm-files pune.osm -o pune.net.xml --geometry.remove ^
  --roundabouts.guess --junctions.join --tls.guess-signals --tls.discard-simple
python run_sumo.py --out-dir data_pune --net pune.net.xml --flow 4 --initial 1500
```

## Things to know

* **Small steps will not speed up.** A step with 1-3 thousand cars clusters
  in about a millisecond, so MPI overhead can make the parallel version
  slower. Speedup shows up with tens of thousands of cars per step. Report
  both cases; the crossover is a result.
* **Parallel and sequential results can differ slightly**, for example a
  border car that could belong to two clusters. They are not forced to match.
* **The replay is a replay, not live coupling.** SUMO and DBSCAN run first;
  the display uses the recorded, real timings.
* While the sequential pass runs on rank 0, the other ranks wait in a
  barrier. If you run more ranks than cores, some MPI builds busy-wait and
  can slow the sequential timing slightly.
* Tune `eps_pos`, `eps_vel` and `minpts` to your scenario. Real SUMO cars
  are usually sparser than the synthetic ones, so you may need
  `eps_pos` 30-40 and `minpts` 3.
