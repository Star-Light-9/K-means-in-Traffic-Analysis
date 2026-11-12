/*
 * dbscan.c - sequential and MPI-parallel DBSCAN on per-step vehicle data.
 *
 * Input : trajectories.csv with header  step,id,x,y,vx,vy,speed
 *         (rows grouped by step)
 * Output: <outdir>/clusters_seq.csv   step,id,cluster
 *         <outdir>/clusters_par.csv   step,id,cluster
 *         <outdir>/timings.csv        per-step timings of both versions
 *
 * Two vehicles are neighbours when
 *     position distance        <= EPS_POS  (metres)   AND
 *     velocity-vector distance <= EPS_VEL  (m/s)
 * Using the velocity VECTOR (vx, vy) means cars heading in opposite
 * directions never share a cluster, and there is no 359-deg / 1-deg
 * angle wrap-around problem.
 *
 * Parallel scheme (per time step):
 *   1. every rank holds the whole step (the vehicle feed is broadcast once)
 *   2. the map is cut into strips along its longer axis, balanced by
 *      vehicle COUNT (BALANCE=1) or by equal width (BALANCE=0)
 *   3. each rank takes its strip plus a halo of width EPS_POS and runs
 *      DBSCAN locally (clusters are only started from owned points)
 *   4. rank 0 gathers the labels and merges local clusters that touch
 *      across strip borders with union-find
 *
 * Usage:
 *   mpirun -np P ./dbscan trajectories.csv outdir [eps_pos] [eps_vel]
 *                         [minpts] [balance] [reps] [max_frames]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define NOISE (-1)
#define UNVISITED (-2)
#define NBINS 8192

static double EPS_POS = 25.0;
static double EPS_VEL = 4.0;
static int MINPTS = 4;
static int BALANCE = 1;

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static void ensure(void **p, size_t *cap, size_t need, size_t elem)
{
    if (need <= *cap)
        return;

    size_t nc = *cap ? *cap : 1024;
    while (nc < need)
        nc *= 2;

    void *q = realloc(*p, nc * elem);

    if (q == NULL)
    {
        fprintf(stderr, "Out of memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    *p = q;
    *cap = nc;
}

/* ------------------------------------------------------------------ */
/* DBSCAN core (used by both the sequential and the parallel version)  */
/* ------------------------------------------------------------------ */

static int *g_head = NULL, *g_next = NULL;
static int *g_nb = NULL, *g_nb2 = NULL, *g_queue = NULL;
static size_t c_head = 0, c_next = 0, c_nb = 0, c_nb2 = 0, c_queue = 0;

static double g_minx, g_miny, g_cell;
static int g_gx, g_gy;
static const double *g_x, *g_y, *g_vx, *g_vy;

static int region_query(int i, int *out)
{
    int cx = (int)((g_x[i] - g_minx) / g_cell);
    int cy = (int)((g_y[i] - g_miny) / g_cell);

    double e2 = EPS_POS * EPS_POS;
    double v2 = EPS_VEL * EPS_VEL;
    int cnt = 0;

    for (int yy = cy - 1; yy <= cy + 1; yy++)
    {
        if (yy < 0 || yy >= g_gy)
            continue;

        for (int xx = cx - 1; xx <= cx + 1; xx++)
        {
            if (xx < 0 || xx >= g_gx)
                continue;

            for (int j = g_head[yy * g_gx + xx]; j >= 0; j = g_next[j])
            {
                double dx = g_x[j] - g_x[i];
                double dy = g_y[j] - g_y[i];

                if (dx * dx + dy * dy > e2)
                    continue;

                double dvx = g_vx[j] - g_vx[i];
                double dvy = g_vy[j] - g_vy[i];

                if (dvx * dvx + dvy * dvy > v2)
                    continue;

                out[cnt++] = j;
            }
        }
    }

    return cnt;
}

/*
 * Points 0..n_owned-1 may start clusters; points n_owned..n-1 are halo
 * ("ghost") points that can only be absorbed into clusters.
 * label[i] : cluster id >= 0, NOISE, or UNVISITED (ghost never reached)
 * core[i]  : 1 if the point was found to be a core point
 * Returns the number of clusters.
 */
static int dbscan_core(
    const double *x, const double *y,
    const double *vx, const double *vy,
    int n, int n_owned,
    int *label, unsigned char *core)
{
    if (n <= 0)
        return 0;

    g_x = x; g_y = y; g_vx = vx; g_vy = vy;

    double minx = x[0], maxx = x[0], miny = y[0], maxy = y[0];

    for (int i = 1; i < n; i++)
    {
        if (x[i] < minx) minx = x[i];
        if (x[i] > maxx) maxx = x[i];
        if (y[i] < miny) miny = y[i];
        if (y[i] > maxy) maxy = y[i];
    }

    double cell = EPS_POS;
    int gx, gy;

    for (;;)
    {
        gx = (int)((maxx - minx) / cell) + 1;
        gy = (int)((maxy - miny) / cell) + 1;

        if ((long long)gx * gy <= 4000000LL)
            break;

        cell *= 2.0;
    }

    g_minx = minx; g_miny = miny; g_cell = cell;
    g_gx = gx; g_gy = gy;

    ensure((void **)&g_head, &c_head, (size_t)gx * gy, sizeof(int));
    ensure((void **)&g_next, &c_next, (size_t)n, sizeof(int));
    ensure((void **)&g_nb, &c_nb, (size_t)n, sizeof(int));
    ensure((void **)&g_nb2, &c_nb2, (size_t)n, sizeof(int));
    ensure((void **)&g_queue, &c_queue, (size_t)n, sizeof(int));

    for (int c = 0; c < gx * gy; c++)
        g_head[c] = -1;

    for (int i = 0; i < n; i++)
    {
        int cx = (int)((x[i] - minx) / cell);
        int cy = (int)((y[i] - miny) / cell);
        int c = cy * gx + cx;

        g_next[i] = g_head[c];
        g_head[c] = i;

        label[i] = UNVISITED;
        core[i] = 0;
    }

    int cid = 0;

    for (int i = 0; i < n_owned; i++)
    {
        if (label[i] != UNVISITED)
            continue;

        int cnt = region_query(i, g_nb);

        if (cnt < MINPTS)
        {
            label[i] = NOISE;
            continue;
        }

        core[i] = 1;
        label[i] = cid;

        int qh = 0, qt = 0;

        for (int k = 0; k < cnt; k++)
        {
            int j = g_nb[k];

            if (label[j] == UNVISITED)
            {
                label[j] = cid;
                g_queue[qt++] = j;
            }
            else if (label[j] == NOISE)
                label[j] = cid;
        }

        while (qh < qt)
        {
            int j = g_queue[qh++];
            int c2 = region_query(j, g_nb2);

            if (c2 < MINPTS)
                continue;

            core[j] = 1;

            for (int k = 0; k < c2; k++)
            {
                int m = g_nb2[k];

                if (label[m] == UNVISITED)
                {
                    label[m] = cid;
                    g_queue[qt++] = m;
                }
                else if (label[m] == NOISE)
                    label[m] = cid;
            }
        }

        cid++;
    }

    return cid;
}

/* ------------------------------------------------------------------ */
/* parallel DBSCAN for one time step                                   */
/* ------------------------------------------------------------------ */

static int *p_own = NULL, *p_gho = NULL, *p_gidx = NULL, *p_lab = NULL;
static double *p_lx = NULL, *p_ly = NULL, *p_lvx = NULL, *p_lvy = NULL;
static unsigned char *p_core = NULL;
static int *p_rec = NULL, *p_recv = NULL;
static int *r_own = NULL, *r_anchor = NULL, *r_ghost = NULL;
static int *r_parent = NULL, *r_comp = NULL;
static size_t cp_own = 0, cp_gho = 0, cp_gidx = 0, cp_lab = 0;
static size_t cp_lx = 0, cp_ly = 0, cp_lvx = 0, cp_lvy = 0, cp_core = 0;
static size_t cp_rec = 0, cp_recv = 0;
static size_t cr_own = 0, cr_anchor = 0, cr_ghost = 0;
static size_t cr_parent = 0, cr_comp = 0;

static int uf_find(int a)
{
    while (r_parent[a] != a)
    {
        r_parent[a] = r_parent[r_parent[a]];
        a = r_parent[a];
    }
    return a;
}

static void uf_union(int a, int b)
{
    a = uf_find(a);
    b = uf_find(b);

    if (a != b)
        r_parent[b] = a;
}

/*
 * Returns on every rank. On rank 0: labels[] (size n) and
 * *nclusters_out are filled and *t_total is the wall time of the whole
 * step (partition + local DBSCAN + gather + merge).
 * On all ranks *t_local is the time until the local DBSCAN finished.
 */
static void par_frame(
    const double *X, const double *Y,
    const double *VX, const double *VY,
    int n, int rank, int size,
    int *labels, int *nclusters_out,
    double *t_total, double *t_local)
{
    double t0 = MPI_Wtime();

    /* ---- choose the split axis and strip borders ---- */
    double minx = X[0], maxx = X[0], miny = Y[0], maxy = Y[0];

    for (int i = 1; i < n; i++)
    {
        if (X[i] < minx) minx = X[i];
        if (X[i] > maxx) maxx = X[i];
        if (Y[i] < miny) miny = Y[i];
        if (Y[i] > maxy) maxy = Y[i];
    }

    const double *C = ((maxx - minx) >= (maxy - miny)) ? X : Y;
    double lo = (C == X) ? minx : miny;
    double hi = (C == X) ? maxx : maxy;

    double *b = malloc((size_t)(size + 1) * sizeof(double));

    b[0] = lo;
    b[size] = hi + 1.0;

    if (!BALANCE || hi <= lo)
    {
        for (int r = 1; r < size; r++)
            b[r] = lo + (hi - lo) * r / size;
    }
    else
    {
        static int hist[NBINS];
        memset(hist, 0, sizeof(hist));

        double width = (hi - lo) / NBINS;

        for (int i = 0; i < n; i++)
        {
            int bin = (int)((C[i] - lo) / width);

            if (bin >= NBINS)
                bin = NBINS - 1;

            hist[bin]++;
        }

        long long cum = 0;
        int bin = 0;

        for (int r = 1; r < size; r++)
        {
            long long target = (long long)r * n / size;

            while (bin < NBINS - 1 && cum + hist[bin] < target)
            {
                cum += hist[bin];
                bin++;
            }

            b[r] = lo + (bin + 1) * width;
        }
    }

    double my_lo = b[rank];
    double my_hi = b[rank + 1];

    /* ---- select owned points and halo points ---- */
    ensure((void **)&p_own, &cp_own, (size_t)n, sizeof(int));
    ensure((void **)&p_gho, &cp_gho, (size_t)n, sizeof(int));

    int no = 0, ng = 0;

    for (int i = 0; i < n; i++)
    {
        double c = C[i];

        if (c >= my_lo && (c < my_hi || rank == size - 1))
            p_own[no++] = i;
        else if (c >= my_lo - EPS_POS && c < my_hi + EPS_POS)
            p_gho[ng++] = i;
    }

    int nl = no + ng;

    ensure((void **)&p_gidx, &cp_gidx, (size_t)nl + 1, sizeof(int));
    ensure((void **)&p_lab, &cp_lab, (size_t)nl + 1, sizeof(int));
    ensure((void **)&p_lx, &cp_lx, (size_t)nl + 1, sizeof(double));
    ensure((void **)&p_ly, &cp_ly, (size_t)nl + 1, sizeof(double));
    ensure((void **)&p_lvx, &cp_lvx, (size_t)nl + 1, sizeof(double));
    ensure((void **)&p_lvy, &cp_lvy, (size_t)nl + 1, sizeof(double));
    ensure((void **)&p_core, &cp_core, (size_t)nl + 1, 1);

    for (int k = 0; k < nl; k++)
    {
        int g = (k < no) ? p_own[k] : p_gho[k - no];

        p_gidx[k] = g;
        p_lx[k] = X[g];
        p_ly[k] = Y[g];
        p_lvx[k] = VX[g];
        p_lvy[k] = VY[g];
    }

    /* ---- local DBSCAN ---- */
    int nclus = dbscan_core(
        p_lx, p_ly, p_lvx, p_lvy, nl, no, p_lab, p_core);

    *t_local = MPI_Wtime() - t0;

    /* ---- records to send: all owned points, labelled ghost points ---- */
    ensure((void **)&p_rec, &cp_rec, (size_t)nl * 3 + 3, sizeof(int));

    int nrec = 0;

    for (int k = 0; k < nl; k++)
    {
        if (k < no)
        {
            p_rec[3 * nrec] = p_gidx[k];
            p_rec[3 * nrec + 1] = p_lab[k] >= 0 ? p_lab[k] : NOISE;
            p_rec[3 * nrec + 2] = 1 | (p_core[k] ? 2 : 0);
            nrec++;
        }
        else if (p_lab[k] >= 0)
        {
            p_rec[3 * nrec] = p_gidx[k];
            p_rec[3 * nrec + 1] = p_lab[k];
            p_rec[3 * nrec + 2] = 0;
            nrec++;
        }
    }

    /* ---- gather ---- */
    int info[2] = { nrec, nclus };
    int *infoall = NULL;
    int *rcounts = NULL, *rdispls = NULL;

    if (rank == 0)
    {
        infoall = malloc((size_t)size * 2 * sizeof(int));
        rcounts = malloc((size_t)size * sizeof(int));
        rdispls = malloc((size_t)size * sizeof(int));
    }

    MPI_Gather(info, 2, MPI_INT, infoall, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        size_t total_rec = 0;

        for (int r = 0; r < size; r++)
        {
            rcounts[r] = infoall[2 * r] * 3;
            rdispls[r] = (int)(total_rec * 3);
            total_rec += (size_t)infoall[2 * r];
        }

        ensure((void **)&p_recv, &cp_recv, total_rec * 3 + 3, sizeof(int));
    }

    MPI_Gatherv(
        p_rec, nrec * 3, MPI_INT,
        p_recv, rcounts, rdispls, MPI_INT,
        0, MPI_COMM_WORLD);

    /* ---- merge on rank 0 ---- */
    if (rank == 0)
    {
        size_t total_cl = 0;
        int *coff = malloc((size_t)size * sizeof(int));

        for (int r = 0; r < size; r++)
        {
            coff[r] = (int)total_cl;
            total_cl += (size_t)infoall[2 * r + 1];
        }

        ensure((void **)&r_parent, &cr_parent, total_cl + 1, sizeof(int));
        ensure((void **)&r_comp, &cr_comp, total_cl + 1, sizeof(int));
        ensure((void **)&r_own, &cr_own, (size_t)n, sizeof(int));
        ensure((void **)&r_anchor, &cr_anchor, (size_t)n, sizeof(int));
        ensure((void **)&r_ghost, &cr_ghost, (size_t)n, sizeof(int));

        for (size_t k = 0; k < total_cl; k++)
        {
            r_parent[k] = (int)k;
            r_comp[k] = -1;
        }

        for (int g = 0; g < n; g++)
        {
            r_own[g] = -1;
            r_anchor[g] = -1;
            r_ghost[g] = -1;
        }

        /* pass 1: owner records */
        for (int r = 0; r < size; r++)
        {
            int *rec = p_recv + rdispls[r];

            for (int k = 0; k < infoall[2 * r]; k++)
            {
                int g = rec[3 * k];
                int lab = rec[3 * k + 1];
                int fl = rec[3 * k + 2];

                if (fl & 1)
                {
                    int key = lab >= 0 ? coff[r] + lab : -1;

                    r_own[g] = key;

                    /* a core point owned by a rank has its full
                       neighbourhood, so its cluster is reliable */
                    if ((fl & 2) && key >= 0)
                        r_anchor[g] = key;
                }
            }
        }

        /* pass 2: halo records -> union with the owner's cluster */
        for (int r = 0; r < size; r++)
        {
            int *rec = p_recv + rdispls[r];

            for (int k = 0; k < infoall[2 * r]; k++)
            {
                int g = rec[3 * k];
                int lab = rec[3 * k + 1];
                int fl = rec[3 * k + 2];

                if ((fl & 1) || lab < 0)
                    continue;

                int key = coff[r] + lab;

                if (r_anchor[g] >= 0)
                    uf_union(r_anchor[g], key);

                if (r_ghost[g] < 0)
                    r_ghost[g] = key;
            }
        }

        /* final labels, numbered in order of first appearance */
        int next = 0;

        for (int g = 0; g < n; g++)
        {
            int key = r_own[g] >= 0 ? r_own[g] : r_ghost[g];

            if (key < 0)
            {
                labels[g] = NOISE;
                continue;
            }

            int root = uf_find(key);

            if (r_comp[root] < 0)
                r_comp[root] = next++;

            labels[g] = r_comp[root];
        }

        *nclusters_out = next;
        *t_total = MPI_Wtime() - t0;

        free(coff);
        free(infoall);
        free(rcounts);
        free(rdispls);
    }

    free(b);
}

/* ------------------------------------------------------------------ */
/* input                                                               */
/* ------------------------------------------------------------------ */

static int load_csv(
    const char *path,
    int **steps_out, int **ids_out,
    double **x_out, double **y_out, double **vx_out, double **vy_out)
{
    FILE *fp = fopen(path, "r");

    if (fp == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    char line[1024];

    if (fgets(line, sizeof(line), fp) == NULL)
    {
        fclose(fp);
        return -1;
    }

    size_t cap = 100000;
    int n = 0;

    int *st = malloc(cap * sizeof(int));
    int *id = malloc(cap * sizeof(int));
    double *x = malloc(cap * sizeof(double));
    double *y = malloc(cap * sizeof(double));
    double *vx = malloc(cap * sizeof(double));
    double *vy = malloc(cap * sizeof(double));

    while (fgets(line, sizeof(line), fp))
    {
        int s, i;
        double px, py, pvx, pvy, spd;

        if (sscanf(line, "%d,%d,%lf,%lf,%lf,%lf,%lf",
                   &s, &i, &px, &py, &pvx, &pvy, &spd) < 6)
            continue;

        if ((size_t)n == cap)
        {
            cap *= 2;
            st = realloc(st, cap * sizeof(int));
            id = realloc(id, cap * sizeof(int));
            x = realloc(x, cap * sizeof(double));
            y = realloc(y, cap * sizeof(double));
            vx = realloc(vx, cap * sizeof(double));
            vy = realloc(vy, cap * sizeof(double));

            if (!st || !id || !x || !y || !vx || !vy)
            {
                fclose(fp);
                return -1;
            }
        }

        st[n] = s; id[n] = i;
        x[n] = px; y[n] = py; vx[n] = pvx; vy[n] = pvy;
        n++;
    }

    fclose(fp);

    *steps_out = st; *ids_out = id;
    *x_out = x; *y_out = y; *vx_out = vx; *vy_out = vy;

    return n;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *input = argc > 1 ? argv[1] : "trajectories.csv";
    const char *outdir = argc > 2 ? argv[2] : ".";

    if (argc > 3) EPS_POS = atof(argv[3]);
    if (argc > 4) EPS_VEL = atof(argv[4]);
    if (argc > 5) MINPTS = atoi(argv[5]);
    if (argc > 6) BALANCE = atoi(argv[6]);

    int reps = argc > 7 ? atoi(argv[7]) : 1;
    int max_frames = argc > 8 ? atoi(argv[8]) : 0;

    if (reps < 1) reps = 1;

    int *steps = NULL, *ids = NULL;
    double *X = NULL, *Y = NULL, *VX = NULL, *VY = NULL;
    int meta[2] = { 0, 0 };   /* rows, frames */
    int *frame_start = NULL, *frame_step = NULL;

    if (rank == 0)
    {
        int n = load_csv(input, &steps, &ids, &X, &Y, &VX, &VY);

        if (n <= 0)
        {
            fprintf(stderr, "No usable rows in %s\n", input);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* split into frames by step number */
        int nf = 0;
        int cap = 1024;

        frame_start = malloc((size_t)(cap + 1) * sizeof(int));
        frame_step = malloc((size_t)cap * sizeof(int));

        for (int i = 0; i < n; i++)
        {
            if (i == 0 || steps[i] != steps[i - 1])
            {
                if (max_frames > 0 && nf == max_frames)
                {
                    n = i;
                    break;
                }

                if (nf == cap)
                {
                    cap *= 2;
                    frame_start = realloc(frame_start, (size_t)(cap + 1) * sizeof(int));
                    frame_step = realloc(frame_step, (size_t)cap * sizeof(int));
                }

                frame_start[nf] = i;
                frame_step[nf] = steps[i];
                nf++;
            }
        }

        frame_start[nf] = n;

        meta[0] = n;
        meta[1] = nf;

        printf("\nTraffic clustering: sequential vs MPI DBSCAN\n");
        printf("--------------------------------------------\n");
        printf("Rows: %d   Steps: %d   MPI processes: %d\n", n, nf, size);
        printf("eps_pos=%.1f m  eps_vel=%.1f m/s  minpts=%d  split=%s  reps=%d\n",
               EPS_POS, EPS_VEL, MINPTS,
               BALANCE ? "count-balanced" : "equal-width", reps);
    }

    MPI_Bcast(meta, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int N = meta[0];
    int NF = meta[1];

    if (rank != 0)
    {
        ids = malloc((size_t)N * sizeof(int));
        X = malloc((size_t)N * sizeof(double));
        Y = malloc((size_t)N * sizeof(double));
        VX = malloc((size_t)N * sizeof(double));
        VY = malloc((size_t)N * sizeof(double));
        frame_start = malloc((size_t)(NF + 1) * sizeof(int));
        frame_step = malloc((size_t)NF * sizeof(int));
    }

    MPI_Bcast(frame_start, NF + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(frame_step, NF, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(VX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(VY, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ids, N, MPI_INT, 0, MPI_COMM_WORLD);

    int maxn = 0;

    for (int f = 0; f < NF; f++)
    {
        int c = frame_start[f + 1] - frame_start[f];
        if (c > maxn) maxn = c;
    }

    char path[1024];
    FILE *fseq = NULL, *fpar = NULL;
    double *seq_ms = NULL;
    int *seq_cl = NULL;
    int *lab = malloc((size_t)(maxn + 1) * sizeof(int));
    unsigned char *core = malloc((size_t)(maxn + 1));

    /* ---------------- sequential baseline (rank 0 only) ---------------- */
    if (rank == 0)
    {
        snprintf(path, sizeof(path), "%s/clusters_seq.csv", outdir);
        fseq = fopen(path, "w");

        if (fseq == NULL)
        {
            fprintf(stderr, "Cannot write %s (does the folder exist?)\n", path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fprintf(fseq, "step,id,cluster\n");

        seq_ms = malloc((size_t)NF * sizeof(double));
        seq_cl = malloc((size_t)NF * sizeof(int));

        for (int f = 0; f < NF; f++)
        {
            int s = frame_start[f];
            int n = frame_start[f + 1] - s;
            int nc = 0;

            double t0 = MPI_Wtime();

            for (int r = 0; r < reps; r++)
                nc = dbscan_core(X + s, Y + s, VX + s, VY + s,
                                 n, n, lab, core);

            seq_ms[f] = (MPI_Wtime() - t0) * 1000.0 / reps;
            seq_cl[f] = nc;

            for (int i = 0; i < n; i++)
                fprintf(fseq, "%d,%d,%d\n",
                        frame_step[f], ids[s + i], lab[i] >= 0 ? lab[i] : NOISE);
        }

        fclose(fseq);
        printf("Sequential pass finished.\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------------- parallel pass (all ranks) ---------------- */
    double *par_ms = NULL, *cmax_ms = NULL, *cmean_ms = NULL;
    int *par_cl = NULL;
    double *lc_all = malloc((size_t)size * sizeof(double));

    if (rank == 0)
    {
        snprintf(path, sizeof(path), "%s/clusters_par.csv", outdir);
        fpar = fopen(path, "w");

        if (fpar == NULL)
        {
            fprintf(stderr, "Cannot write %s\n", path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fprintf(fpar, "step,id,cluster\n");

        par_ms = malloc((size_t)NF * sizeof(double));
        cmax_ms = malloc((size_t)NF * sizeof(double));
        cmean_ms = malloc((size_t)NF * sizeof(double));
        par_cl = malloc((size_t)NF * sizeof(int));
    }

    for (int f = 0; f < NF; f++)
    {
        int s = frame_start[f];
        int n = frame_start[f + 1] - s;
        int nc = 0;

        double sum_total = 0.0, sum_max = 0.0, sum_mean = 0.0;

        for (int r = 0; r < reps; r++)
        {
            double t_total = 0.0, t_local = 0.0;

            MPI_Barrier(MPI_COMM_WORLD);

            par_frame(X + s, Y + s, VX + s, VY + s, n, rank, size,
                      lab, &nc, &t_total, &t_local);

            /* load-balance statistics (not part of the timed region) */
            MPI_Gather(&t_local, 1, MPI_DOUBLE,
                       lc_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                double mx = 0.0, sm = 0.0;

                for (int p = 0; p < size; p++)
                {
                    if (lc_all[p] > mx) mx = lc_all[p];
                    sm += lc_all[p];
                }

                sum_total += t_total;
                sum_max += mx;
                sum_mean += sm / size;
            }
        }

        if (rank == 0)
        {
            par_ms[f] = sum_total * 1000.0 / reps;
            cmax_ms[f] = sum_max * 1000.0 / reps;
            cmean_ms[f] = sum_mean * 1000.0 / reps;
            par_cl[f] = nc;

            for (int i = 0; i < n; i++)
                fprintf(fpar, "%d,%d,%d\n",
                        frame_step[f], ids[s + i], lab[i] >= 0 ? lab[i] : NOISE);
        }
    }

    /* ---------------- timings + summary ---------------- */
    if (rank == 0)
    {
        fclose(fpar);

        snprintf(path, sizeof(path), "%s/timings.csv", outdir);
        FILE *ft = fopen(path, "w");

        if (ft == NULL)
        {
            fprintf(stderr, "Cannot write %s\n", path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fprintf(ft, "step,n,seq_ms,par_ms,par_compute_max_ms,"
                    "par_compute_mean_ms,seq_clusters,par_clusters\n");

        double tot_seq = 0.0, tot_par = 0.0, imb = 0.0;

        for (int f = 0; f < NF; f++)
        {
            int n = frame_start[f + 1] - frame_start[f];

            fprintf(ft, "%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d\n",
                    frame_step[f], n, seq_ms[f], par_ms[f],
                    cmax_ms[f], cmean_ms[f], seq_cl[f], par_cl[f]);

            tot_seq += seq_ms[f];
            tot_par += par_ms[f];

            if (cmean_ms[f] > 0.0)
                imb += cmax_ms[f] / cmean_ms[f];
        }

        fclose(ft);

        double speedup = tot_par > 0.0 ? tot_seq / tot_par : 0.0;

        printf("\nTotal sequential time: %.3f ms\n", tot_seq);
        printf("Total parallel time:   %.3f ms\n", tot_par);
        printf("Speedup:    %.3f\n", speedup);
        printf("Efficiency: %.1f%%\n", 100.0 * speedup / size);
        printf("Load imbalance (slowest / average rank): %.2f\n", imb / NF);
        printf("\nWrote clusters_seq.csv, clusters_par.csv, timings.csv in %s\n",
               outdir);
    }

    free(lab);
    free(core);
    free(lc_all);

    MPI_Finalize();
    return 0;
}
