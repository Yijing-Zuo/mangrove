#include <cstdint>
#include <cstddef>

#ifdef MG_EMU
#include "mg_emu.h"
#else
#include <cuda_runtime.h>
#define MG_DYN_SHARED_PTR() ((void*)mg_dyn_smem)
extern __shared__ unsigned long long mg_dyn_smem[];
#endif

#if defined(_WIN32) && !defined(MG_EMU)
#define MG_EXPORT extern "C" __declspec(dllexport)
#else
#define MG_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define MG_VERSION 4
#define MG_MAX_GROUP 512
#define MG_MAX_THREADS 1024
#ifndef MG_BH_TILE
#define MG_BH_TILE 4
#endif

typedef long long i64;
typedef unsigned long long u64;

__device__ __forceinline__ void atom_add_i64(i64* p, i64 v) { atomicAdd((u64*)p, (u64)v); }
__device__ __forceinline__ void atom_add_i64_as32(i64* p, i64 v) {
    unsigned* lo = (unsigned*)p;
    unsigned* hi = lo + 1;
    const unsigned xl = (unsigned)(u64)v;
    const unsigned xh = (unsigned)(((u64)v) >> 32);
    const unsigned old = atomicAdd(lo, xl);
    const unsigned carry = (old > 0xFFFFFFFFu - xl) ? 1u : 0u;
    atomicAdd(hi, xh + carry);
}
__device__ __forceinline__ void atom_add_i64_as32_nocarry(i64* p, i64 v) {
    unsigned* lo = (unsigned*)p;
    atomicAdd(lo, (unsigned)(u64)v);
    atomicAdd(lo + 1, (unsigned)(((u64)v) >> 32));
}

__device__ __forceinline__ int upper_bound_i64(const i64* __restrict__ cum, int n, i64 v) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (cum[mid] > v) hi = mid; else lo = mid + 1; }
    return lo;
}
__device__ __forceinline__ i64 lower_bound_i64(const i64* __restrict__ a, i64 n, i64 key) {
    i64 lo = 0, hi = n;
    while (lo < hi) { i64 mid = (lo + hi) >> 1; if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}
__device__ __forceinline__ int upper_bound_i32(const int* __restrict__ a, int n, int v) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] > v) hi = mid; else lo = mid + 1; }
    return lo;
}
__device__ __forceinline__ void chunk_rows(const i64* __restrict__ cum_ch, int nb, i64 ch, const int32_t* __restrict__ beg,
                                           const int32_t* __restrict__ end, int rch, int* s_out, int* p0, int* p1) {
    int s = upper_bound_i64(cum_ch, nb, ch);
    i64 cexcl = s ? cum_ch[s - 1] : 0;
    int c = (int)(ch - cexcl);
    int b = beg[s], e = end[s];
    int q0 = b + c * rch;
    int q1 = q0 + rch; if (q1 > e) q1 = e;
    *s_out = s; *p0 = q0; *p1 = q1;
}
__device__ __forceinline__ void split_word(i64 v, i64* G, i64* H) {
    i64 h = v & 0xFFFFFFFFLL;
    *H = h;
    *G = (v - h) >> 32;
}
__device__ __forceinline__ int block_excl_scan(int v, int* sc, int* total) {
    const int n = (int)blockDim.x, t = (int)threadIdx.x;
    int in = 0;
    sc[t] = v;
    __syncthreads();
    for (int off = 1; off < n; off <<= 1) {
        int cur = sc[in * n + t];
        int add = (t >= off) ? sc[in * n + t - off] : 0;
        sc[(1 - in) * n + t] = cur + add;
        in ^= 1;
        __syncthreads();
    }
    int incl = sc[in * n + t];
    *total = sc[in * n + n - 1];
    __syncthreads();
    return incl - v;
}

struct BHArgs {
    const uint8_t* X; const int32_t* xrow; const i64* gh; const int32_t* wl;
    const int32_t* beg; const int32_t* end; const int32_t* dest; const uint8_t* fmask;
    i64* out; i64 plane;
    const i64* cum_tiny;
    const i64* cum_ch;
    const int32_t* fgs;
    const int32_t* fbo;
    int nb, d, B, ng, rch, wide;
    i64 n_tiny_rows; int n_tiny_blocks; int smem_cells;
    int narrow;
};
__device__ __forceinline__ int bh_planes(const BHArgs& a) { return (a.wide && !a.narrow) ? 2 : 1; }

__global__ void build_hist_kernel(BHArgs a) {
    const int bid = (int)blockIdx.x;
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    const int d = a.d, B = a.B;
    if (bid < a.n_tiny_blocks) {
        const i64 r0 = (i64)bid * a.rch;
        i64 r1 = r0 + a.rch; if (r1 > a.n_tiny_rows) r1 = a.n_tiny_rows;
        for (i64 r = r0 + tid; r < r1; r += bdim) {
            const int s = upper_bound_i64(a.cum_tiny, a.nb, r);
            const i64 excl = s ? a.cum_tiny[s - 1] : 0;
            const int pos = a.beg[s] + (int)(r - excl);
            const int q = a.wl[pos];
            const i64 v = a.gh[q];
            const uint8_t* xr = a.X + (size_t)a.xrow[q] * d;
            const uint8_t* fm = a.fmask + (size_t)s * d;
            i64* o = a.out + (size_t)a.dest[s] * d * B;
            if (a.wide) {
                i64 G, H; split_word(v, &G, &H);
                for (int f = 0; f < d; ++f) {
                    if (!fm[f]) continue;
                    const size_t idx = (size_t)f * B + xr[f];
                    atom_add_i64(o + idx, G);
                    atom_add_i64(o + a.plane + idx, H);
                }
            } else {
                for (int f = 0; f < d; ++f) {
                    if (!fm[f]) continue;
                    atom_add_i64(o + (size_t)f * B + xr[f], v);
                }
            }
        }
        return;
    }
    __shared__ uint8_t fm_s[MG_MAX_GROUP];
    __shared__ int off_s[MG_MAX_GROUP + 1];
    __shared__ int nmask_s;
    i64* sh = (i64*)MG_DYN_SHARED_PTR();
    const int item = bid - a.n_tiny_blocks;
    const int ch = item / a.ng;
    const int g = item - ch * a.ng;
    int s, p0, p1;
    chunk_rows(a.cum_ch, a.nb, ch, a.beg, a.end, a.rch, &s, &p0, &p1);
    const int f0 = a.fgs[g], f1 = a.fgs[g + 1], F = f1 - f0;
    const int base = a.fbo[f0];
    const int width = a.fbo[f1] - base;
    const int planes = bh_planes(a);
    const bool two_planes = planes == 2;
    for (int j = tid; j <= F; j += bdim) {
        off_s[j] = a.fbo[f0 + j] - base;
        if (j < F) fm_s[j] = a.fmask[(size_t)s * d + f0 + j];
    }
    int R = a.smem_cells / (width * planes);
    const int nwarps = bdim >> 5;
    if (R > nwarps) R = nwarps;
    if (R > 8) R = 8;
    if (R < 1) R = 1;
    const int tw = width * planes;
    for (int i = tid; i < tw * R; i += bdim) sh[i] = 0;
    if (tid == 0) nmask_s = 0;
    __syncthreads();
    for (int j = tid; j < F; j += bdim) if (!fm_s[j]) atomicAdd(&nmask_s, 1);
    __syncthreads();
    const bool allon = nmask_s == 0;
    i64* my = sh + ((tid >> 5) % R) * tw;
    for (int pos = p0 + tid; pos < p1; pos += bdim * MG_BH_TILE) {
        const uint8_t* xr[MG_BH_TILE];
        i64 v[MG_BH_TILE];
#pragma unroll
        for (int t = 0; t < MG_BH_TILE; ++t) {
            const int pp = pos + t * bdim;
            if (pp < p1) {
                const int q = a.wl[pp];
                v[t] = a.gh[q];
                xr[t] = a.X + (size_t)a.xrow[q] * d + f0;
            } else {
                v[t] = 0;
                xr[t] = nullptr;
            }
        }
        if (two_planes) {
            i64 G[MG_BH_TILE], H[MG_BH_TILE];
#pragma unroll
            for (int t = 0; t < MG_BH_TILE; ++t) split_word(v[t], &G[t], &H[t]);
            for (int j = 0; j < F; ++j) {
                if (!allon && !fm_s[j]) continue;
                const int off = off_s[j];
#pragma unroll
                for (int t = 0; t < MG_BH_TILE; ++t) {
                    if (xr[t] == nullptr) continue;
                    const int idx = off + xr[t][j];
                    atom_add_i64_as32(my + idx, G[t]);
                    atom_add_i64_as32(my + width + idx, H[t]);
                }
            }
        } else {
            if (a.narrow) {
                for (int j = 0; j < F; ++j) {
                    if (!allon && !fm_s[j]) continue;
                    const int off = off_s[j];
#pragma unroll
                    for (int t = 0; t < MG_BH_TILE; ++t) {
                        if (xr[t] == nullptr) continue;
                        atom_add_i64_as32_nocarry(my + off + xr[t][j], v[t]);
                    }
                }
            } else {
                for (int j = 0; j < F; ++j) {
                    if (!allon && !fm_s[j]) continue;
                    const int off = off_s[j];
#pragma unroll
                    for (int t = 0; t < MG_BH_TILE; ++t) {
                        if (xr[t] == nullptr) continue;
                        atom_add_i64_as32(my + off + xr[t][j], v[t]);
                    }
                }
            }
        }
    }
    __syncthreads();
    i64* o = a.out + (size_t)a.dest[s] * d * B;
    for (int idx = tid; idx < width; idx += bdim) {
        const int j = upper_bound_i32(off_s + 1, F, idx);
        if (!fm_s[j]) continue;
        const int b = idx - off_s[j];
        const size_t oidx = (size_t)(f0 + j) * B + b;
        if (a.wide) {
            i64 vg = 0, vh = 0;
            if (two_planes) {
                for (int r = 0; r < R; ++r) { vg += sh[r * tw + idx]; vh += sh[r * tw + width + idx]; }
            } else {
                i64 v = 0;
                for (int r = 0; r < R; ++r) v += sh[r * tw + idx];
                split_word(v, &vg, &vh);
            }
            if (vg) atom_add_i64(o + oidx, vg);
            if (vh) atom_add_i64(o + a.plane + oidx, vh);
        } else {
            i64 v = 0;
            for (int r = 0; r < R; ++r) v += sh[r * tw + idx];
            if (v) atom_add_i64(o + oidx, v);
        }
    }
}

__global__ void node_sums_kernel(const i64* __restrict__ gh, const int32_t* __restrict__ wl, const int32_t* __restrict__ beg,
                                 const int32_t* __restrict__ end, const i64* __restrict__ cum_ch, int nb, int rch,
                                 i64* __restrict__ G, i64* __restrict__ H) {
    __shared__ i64 sG[MG_MAX_THREADS];
    __shared__ i64 sH[MG_MAX_THREADS];
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    int s, p0, p1;
    chunk_rows(cum_ch, nb, (i64)blockIdx.x, beg, end, rch, &s, &p0, &p1);
    i64 g = 0, h = 0;
    for (int pos = p0 + tid; pos < p1; pos += bdim) {
        i64 vg, vh; split_word(gh[wl[pos]], &vg, &vh);
        g += vg; h += vh;
    }
    sG[tid] = g; sH[tid] = h;
    __syncthreads();
    if (tid == 0) {
        i64 tg = 0, th = 0;
        for (int i = 0; i < bdim; ++i) { tg += sG[i]; th += sH[i]; }
        atom_add_i64(G + s, tg);
        atom_add_i64(H + s, th);
    }
}

__global__ void leaf_update_kernel(float* __restrict__ pred, const int32_t* __restrict__ wl, const int32_t* __restrict__ beg,
                                   const int32_t* __restrict__ end, const i64* __restrict__ cum_ch, const float* __restrict__ lv,
                                   int nb, int rch) {
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    int s, p0, p1;
    chunk_rows(cum_ch, nb, (i64)blockIdx.x, beg, end, rch, &s, &p0, &p1);
    const float v = lv[s];
    for (int pos = p0 + tid; pos < p1; pos += bdim) {
        const int q = wl[pos];
        pred[q] = pred[q] + v;
    }
}

__global__ void part_count_kernel(const uint8_t* __restrict__ X, const int32_t* __restrict__ xrow, const int32_t* __restrict__ wl,
                                  const int32_t* __restrict__ beg, const int32_t* __restrict__ end, const int32_t* __restrict__ feat,
                                  const int32_t* __restrict__ thr, const i64* __restrict__ cum_ch,
                                  const i64* __restrict__ cum_len, int P, int d, int rch,
                                  int32_t* __restrict__ cnt, uint8_t* __restrict__ fl) {
    __shared__ int tot;
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    int s, p0, p1;
    chunk_rows(cum_ch, P, (i64)blockIdx.x, beg, end, rch, &s, &p0, &p1);
    if (tid == 0) tot = 0;
    __syncthreads();
    const int f = feat[s], t = thr[s];
    const i64 fbase = (s ? cum_len[s - 1] : 0) - beg[s];
    int c = 0;
    for (int pos = p0 + tid; pos < p1; pos += bdim) {
        const int q = wl[pos];
        const uint8_t left = ((int)X[(size_t)xrow[q] * d + f] <= t) ? 1 : 0;
        fl[fbase + pos] = left;
        c += left;
    }
    atomicAdd(&tot, c);
    __syncthreads();
    if (tid == 0) cnt[blockIdx.x] = tot;
}

__global__ void part_scatter_kernel(const int32_t* __restrict__ wl,
                                    const int32_t* __restrict__ beg, const int32_t* __restrict__ end,
                                    const i64* __restrict__ cum_ch, const i64* __restrict__ cum_len,
                                    int P, int rch, const int32_t* __restrict__ cnt, const uint8_t* __restrict__ fl,
                                    int32_t* __restrict__ tmp, int32_t* __restrict__ mid) {
    __shared__ int sc[2 * MG_MAX_THREADS];
    __shared__ int s_before, s_nleft;
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    const i64 ch = (i64)blockIdx.x;
    const int s = upper_bound_i64(cum_ch, P, ch);
    const i64 cexcl = s ? cum_ch[s - 1] : 0;
    const int c = (int)(ch - cexcl);
    const int b = beg[s], e = end[s];
    int p0 = b + c * rch, p1 = p0 + rch; if (p1 > e) p1 = e;
    if (tid == 0) {
        int before = 0, nl = 0;
        const int nch = (int)(cum_ch[s] - cexcl);
        for (int k = 0; k < nch; ++k) { const int v = cnt[cexcl + k]; nl += v; if (k < c) before += v; }
        s_before = before; s_nleft = nl;
        if (c == 0) mid[s] = b + nl;
    }
    __syncthreads();
    const i64 tbase = s ? cum_len[s - 1] : 0;
    int run = s_before;
    const int nleft = s_nleft;
    for (int t0 = p0; t0 < p1; t0 += bdim) {
        const int pos = t0 + tid;
        const bool valid = pos < p1;
        int q = 0, left = 0;
        if (valid) { q = wl[pos]; left = fl[tbase + (pos - b)]; }
        int total;
        const int excl = block_excl_scan(left, sc, &total);
        if (valid) {
            const int i = pos - b;
            const int lefts_before = run + excl;
            const int dst = left ? lefts_before : nleft + (i - lefts_before);
            tmp[tbase + dst] = q;
        }
        run += total;
    }
}

__global__ void part_copy_kernel(int32_t* __restrict__ wl, const int32_t* __restrict__ beg, const int32_t* __restrict__ end,
                                 const i64* __restrict__ cum_ch, const i64* __restrict__ cum_len, int P, int rch,
                                 const int32_t* __restrict__ tmp) {
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    int s, p0, p1;
    chunk_rows(cum_ch, P, (i64)blockIdx.x, beg, end, rch, &s, &p0, &p1);
    const i64 tbase = s ? cum_len[s - 1] : 0;
    const int b = beg[s];
    for (int pos = p0 + tid; pos < p1; pos += bdim) wl[pos] = tmp[tbase + (pos - b)];
}

__global__ void route_update_kernel(const uint8_t* __restrict__ X, const int32_t* __restrict__ xrow, float* __restrict__ pred,
                                    const i64* __restrict__ q, const i64* __restrict__ slot_of_q, i64 R,
                                    const int32_t* __restrict__ lev_off, const i64* __restrict__ keys,
                                    const int32_t* __restrict__ feat, const int32_t* __restrict__ thr, int L,
                                    const i64* __restrict__ leaf_key, const float* __restrict__ leaf_val, i64 n_leaf,
                                    int node_shift, int d, int* __restrict__ err) {
    const i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const i64 qi = q[i];
    const i64 sl = slot_of_q[i] << node_shift;
    const uint8_t* xr = X + (size_t)xrow[qi] * d;
    i64 nid = 0;
    for (int l = 0; l < L; ++l) {
        const int o = lev_off[l];
        const i64 n = (i64)(lev_off[l + 1] - o);
        if (n == 0) continue;
        const i64 key = sl | nid;
        i64 j = lower_bound_i64(keys + o, n, key);
        if (j > n - 1) j = n - 1;
        if (keys[o + j] == key) {
            const int f = feat[o + j];
            nid = 2 * nid + 1 + (((int)xr[f] > thr[o + j]) ? 1 : 0);
        }
    }
    const i64 key = sl | nid;
    i64 j = lower_bound_i64(leaf_key, n_leaf, key);
    if (j > n_leaf - 1) j = n_leaf - 1;
    if (n_leaf == 0 || leaf_key[j] != key) { atomicAdd(err, 1); return; }
    pred[qi] = pred[qi] + leaf_val[j];
}

#ifdef MG_EMU
#define MG_LAUNCH(kernel, grid, threads, smem, stream, ...) \
    mg_emu_launch((unsigned)(grid), (unsigned)(threads), (size_t)(smem), [=]() { kernel(__VA_ARGS__); })
#else
#define MG_LAUNCH(kernel, grid, threads, smem, stream, ...) \
    (kernel<<<(unsigned)(grid), (unsigned)(threads), (size_t)(smem), (cudaStream_t)(stream)>>>(__VA_ARGS__), (int)cudaGetLastError())
#endif

MG_EXPORT int mg_version() { return MG_VERSION; }

#ifndef MG_EMU
static int configure_hist_shared_memory(int smem_bytes) {
    if (smem_bytes <= 0) return (int)cudaErrorInvalidValue;
    static thread_local int cached_device = -1, cached_dynamic = 0;
    int device;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) return (int)e;
    if (device == cached_device && smem_bytes <= cached_dynamic) return 0;
    cudaFuncAttributes attrs;
    e = cudaFuncGetAttributes(&attrs, build_hist_kernel);
    if (e != cudaSuccess) return (int)e;
    int max_shared;
    e = cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (e != cudaSuccess) return (int)e;
    if (attrs.sharedSizeBytes + (size_t)smem_bytes > (size_t)max_shared)
        return (int)cudaErrorInvalidValue;
    if (smem_bytes > attrs.maxDynamicSharedSizeBytes) {
        e = cudaFuncSetAttribute(build_hist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (e != cudaSuccess) return (int)e;
    }
    cached_device = device;
    cached_dynamic = smem_bytes;
    return 0;
}
#endif

MG_EXPORT int mg_build_hist(const void* X, const void* xrow, const void* gh, const void* wl, const void* beg, const void* end,
                            const void* dest, const void* fmask, void* out, i64 plane, const void* cum_tiny, const void* cum_ch,
                            const void* fgs, const void* fbo, int nb, int d, int B, int ng, int rch, int wide,
                            i64 n_tiny_rows, int n_tiny_blocks, i64 n_chunks, void* stream, int threads, int smem_bytes,
                            int narrow) {
    BHArgs a;
    a.X = (const uint8_t*)X; a.xrow = (const int32_t*)xrow; a.gh = (const i64*)gh; a.wl = (const int32_t*)wl;
    a.beg = (const int32_t*)beg; a.end = (const int32_t*)end; a.dest = (const int32_t*)dest; a.fmask = (const uint8_t*)fmask;
    a.out = (i64*)out; a.plane = plane; a.cum_tiny = (const i64*)cum_tiny; a.cum_ch = (const i64*)cum_ch;
    a.fgs = (const int32_t*)fgs; a.fbo = (const int32_t*)fbo;
    a.nb = nb; a.d = d; a.B = B; a.ng = ng; a.rch = rch; a.wide = wide;
    a.n_tiny_rows = n_tiny_rows; a.n_tiny_blocks = n_tiny_blocks; a.smem_cells = smem_bytes / 8;
    a.narrow = narrow ? 1 : 0;
    if (a.narrow && rch > 65536) return -5;
    const i64 grid = (i64)n_tiny_blocks + n_chunks * (i64)ng;
    if (grid <= 0) return 0;
    if (grid > 2147483647LL) return -2;
    if (threads < 32 || threads > MG_MAX_THREADS || (threads & 31)) return -3;
    if (smem_bytes <= 0) return -4;
#ifndef MG_EMU
    const int smem_rc = configure_hist_shared_memory(smem_bytes);
    if (smem_rc) return smem_rc;
#endif
    return MG_LAUNCH(build_hist_kernel, grid, threads, smem_bytes, stream, a);
}

MG_EXPORT int mg_node_sums(const void* gh, const void* wl, const void* beg, const void* end, const void* cum_ch, int nb, int rch,
                           i64 n_chunks, void* G, void* H, void* stream, int threads) {
    if (n_chunks <= 0) return 0;
    if (threads < 1 || threads > MG_MAX_THREADS) return -3;
    return MG_LAUNCH(node_sums_kernel, n_chunks, threads, 0, stream, (const i64*)gh, (const int32_t*)wl, (const int32_t*)beg,
                     (const int32_t*)end, (const i64*)cum_ch, nb, rch, (i64*)G, (i64*)H);
}

MG_EXPORT int mg_leaf_update(void* pred, const void* wl, const void* beg, const void* end, const void* cum_ch, const void* lv,
                             int nb, int rch, i64 n_chunks, void* stream, int threads) {
    if (n_chunks <= 0) return 0;
    if (threads < 1 || threads > MG_MAX_THREADS) return -3;
    return MG_LAUNCH(leaf_update_kernel, n_chunks, threads, 0, stream, (float*)pred, (const int32_t*)wl, (const int32_t*)beg,
                     (const int32_t*)end, (const i64*)cum_ch, (const float*)lv, nb, rch);
}

MG_EXPORT int mg_partition(const void* X, const void* xrow, void* wl, const void* beg, const void* end, const void* feat,
                           const void* thr, const void* cum_ch, const void* cum_len, int P, int d, int rch, i64 n_chunks,
                           void* cnt, void* tmp, void* mid, void* stream, int threads, void* fl) {
    if (n_chunks <= 0) return 0;
    if (threads < 1 || threads > MG_MAX_THREADS) return -3;
    int rc = MG_LAUNCH(part_count_kernel, n_chunks, threads, 0, stream, (const uint8_t*)X, (const int32_t*)xrow, (const int32_t*)wl,
                       (const int32_t*)beg, (const int32_t*)end, (const int32_t*)feat, (const int32_t*)thr, (const i64*)cum_ch,
                       (const i64*)cum_len, P, d, rch, (int32_t*)cnt, (uint8_t*)fl);
    if (rc) return rc;
    rc = MG_LAUNCH(part_scatter_kernel, n_chunks, threads, 0, stream, (const int32_t*)wl,
                   (const int32_t*)beg, (const int32_t*)end, (const i64*)cum_ch,
                   (const i64*)cum_len, P, rch, (const int32_t*)cnt, (const uint8_t*)fl, (int32_t*)tmp, (int32_t*)mid);
    if (rc) return rc;
    return MG_LAUNCH(part_copy_kernel, n_chunks, threads, 0, stream, (int32_t*)wl, (const int32_t*)beg, (const int32_t*)end,
                     (const i64*)cum_ch, (const i64*)cum_len, P, rch, (const int32_t*)tmp);
}

MG_EXPORT int mg_route_update(const void* X, const void* xrow, void* pred, const void* q, const void* slot_of_q, i64 R,
                              const void* lev_off, const void* keys, const void* feat, const void* thr, int L,
                              const void* leaf_key, const void* leaf_val, i64 n_leaf, int node_shift, int d, void* err,
                              void* stream, int threads) {
    if (R <= 0) return 0;
    if (threads < 1 || threads > MG_MAX_THREADS) return -3;
    const i64 grid = (R + threads - 1) / threads;
    if (grid > 2147483647LL) return -2;
    return MG_LAUNCH(route_update_kernel, grid, threads, 0, stream, (const uint8_t*)X, (const int32_t*)xrow, (float*)pred,
                     (const i64*)q, (const i64*)slot_of_q, R, (const int32_t*)lev_off, (const i64*)keys, (const int32_t*)feat,
                     (const int32_t*)thr, L, (const i64*)leaf_key, (const float*)leaf_val, n_leaf, node_shift, d, (int*)err);
}

#ifdef MG_EMU
thread_local mg_dim3 threadIdx = {0, 0, 0};
mg_dim3 blockIdx = {0, 0, 0}, blockDim = {1, 1, 1}, gridDim = {1, 1, 1};

namespace {
struct EmuCtx {
    unsigned grid, threads; std::vector<unsigned char> dyn; pthread_barrier_t bar; const std::function<void()>* body;
};
EmuCtx* g_ctx = nullptr;
void* worker(void* arg) {
    const unsigned tid = (unsigned)(uintptr_t)arg;
    threadIdx.x = tid; threadIdx.y = 0; threadIdx.z = 0;
    for (unsigned b = 0; b < g_ctx->grid; ++b) {
        pthread_barrier_wait(&g_ctx->bar);
        if (tid == 0) { blockIdx.x = b; if (!g_ctx->dyn.empty()) memset(g_ctx->dyn.data(), 0xAB, g_ctx->dyn.size()); }
        pthread_barrier_wait(&g_ctx->bar);
        (*g_ctx->body)();
    }
    return nullptr;
}
}

void mg_emu_sync() { pthread_barrier_wait(&g_ctx->bar); }
void* mg_emu_dyn_shared() { return g_ctx->dyn.data(); }

int mg_emu_launch(unsigned grid, unsigned threads, size_t smem_bytes, const std::function<void()>& body) {
    if (grid == 0) return 0;
    if (threads == 0) return -3;
    EmuCtx ctx;
    ctx.grid = grid; ctx.threads = threads; ctx.body = &body;
    ctx.dyn.assign(smem_bytes ? smem_bytes : 8, 0xAB);
    pthread_barrier_init(&ctx.bar, nullptr, threads);
    g_ctx = &ctx;
    blockDim.x = threads; blockDim.y = 1; blockDim.z = 1;
    gridDim.x = grid; gridDim.y = 1; gridDim.z = 1;
    std::vector<pthread_t> th(threads);
    for (unsigned t = 0; t < threads; ++t) pthread_create(&th[t], nullptr, worker, (void*)(uintptr_t)t);
    for (unsigned t = 0; t < threads; ++t) pthread_join(th[t], nullptr);
    pthread_barrier_destroy(&ctx.bar);
    g_ctx = nullptr;
    return 0;
}
#endif
