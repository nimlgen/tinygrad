#include <hip/hip_runtime.h>

#ifndef MK_DIM
#define MK_DIM 65536
#endif
#ifndef E_DIM
#define E_DIM 32
#endif
#ifndef BLOCK_ROW
#define BLOCK_ROW 256
#endif
#ifndef THREADS
#define THREADS 256
#endif

// Routing metadata of one group per workgroup, exactly moe_routing.route_topk: per-expert counts, BLOCK_ROW-padded
// offsets and each assignment's grouped row. Thread t owns MK_DIM/THREADS consecutive assignments, so an expert's rows
// keep assignment order: hist[e][t] becomes the number of earlier assignments to e owned by lower threads.
extern "C" __global__ __launch_bounds__(THREADS) void moe_route_meta(
    int *__restrict__ counts,
    int *__restrict__ off,
    int *__restrict__ dest_row,
    const int *__restrict__ topi) {
  static_assert(MK_DIM % THREADS == 0 && E_DIM <= THREADS);
  constexpr int PER = MK_DIM / THREADS;
  __shared__ int hist[E_DIM][THREADS];
  __shared__ int base[E_DIM];
  const int group = blockIdx.x, t = threadIdx.x;
  const int *ids = topi + (long long)group * MK_DIM + t * PER;

  for (int e = 0; e < E_DIM; e++) hist[e][t] = 0;
  for (int j = 0; j < PER; j++) hist[ids[j]][t]++;
  __syncthreads();

  if (t < E_DIM) {
    int run = 0;
    for (int u = 0; u < THREADS; u++) { const int c = hist[t][u]; hist[t][u] = run; run += c; }
    counts[group * E_DIM + t] = run;
    base[t] = (run + BLOCK_ROW - 1) / BLOCK_ROW * BLOCK_ROW;
  }
  __syncthreads();

  if (t == 0) {
    int acc = 0;
    for (int e = 0; e < E_DIM; e++) { const int padded = base[e]; base[e] = acc; off[group * (E_DIM + 1) + e] = acc; acc += padded; }
    off[group * (E_DIM + 1) + E_DIM] = acc;
  }
  __syncthreads();

  int *rows = dest_row + (long long)group * MK_DIM + t * PER;
  for (int j = 0; j < PER; j++) { const int e = ids[j]; rows[j] = base[e] + hist[e][t]++; }
}
