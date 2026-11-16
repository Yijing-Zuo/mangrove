#pragma once
#include <pthread.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <functional>

struct mg_dim3 { unsigned x, y, z; };
extern thread_local mg_dim3 threadIdx;
extern mg_dim3 blockIdx, blockDim, gridDim;

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
#define __syncthreads() mg_emu_sync()
#define MG_DYN_SHARED_PTR() mg_emu_dyn_shared()

void mg_emu_sync();
void* mg_emu_dyn_shared();
int mg_emu_launch(unsigned grid, unsigned threads, size_t smem_bytes, const std::function<void()>& body);

inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline unsigned atomicAdd(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }

typedef void* cudaStream_t;
