#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#if defined(_WIN32)
#define NM_EXPORT extern "C" __declspec(dllexport)
#else
#define NM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define MGNM_VERSION 9
#define NM_MAXB 128
#define NM_MAXFG 32
#define NM_HIST_MAX_THREADS 512
#define NM_MAXK 32
#define NM_GRAD_MAX_GRID (1 << 20)

typedef long long i64;
typedef unsigned long long u64;
extern __shared__ unsigned long long nm_dyn_smem[];

namespace mgnm {

struct Desc {
    const i64* sa; const i64* sb; const i64* cum_ch; const i64* xbase; const i64* slot;
    const i64* eoff; const i64* node; const i64* aux; const i64* aux2;
    int ns, E;
};

static Desc make_desc(const void* p, int ns, int E) {
    const i64* b = (const i64*)p;
    Desc d;
    d.sa = b; d.sb = b + ns; d.cum_ch = b + 2 * (size_t)ns; d.xbase = b + 3 * (size_t)ns; d.slot = b + 4 * (size_t)ns;
    d.eoff = b + 5 * (size_t)ns; d.node = d.eoff + ns + 1; d.aux = d.node + E; d.aux2 = d.aux + E;
    d.ns = ns; d.E = E;
    return d;
}

__device__ __forceinline__ void atom_add_i64(i64* p, i64 v) { atomicAdd((u64*)p, (u64)v); }
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
__device__ __forceinline__ int upper_bound_i32(const int* __restrict__ a, int n, int v) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] > v) hi = mid; else lo = mid + 1; }
    return lo;
}
__device__ __forceinline__ void split_word(i64 v, i64* G, i64* H) {
    i64 h = v & 0xFFFFFFFFLL;
    *H = h;
    *G = (v - h) >> 32;
}
__device__ __forceinline__ void slot_chunk(const Desc& D, i64 ch, int rch, int* i_out, i64* r0, i64* r1) {
    const int i = upper_bound_i64(D.cum_ch, D.ns, ch);
    const i64 c = ch - (i ? D.cum_ch[i - 1] : 0);
    const i64 a = D.sa[i] + c * (i64)rch;
    i64 e = a + rch; if (e > D.sb[i]) e = D.sb[i];
    *i_out = i; *r0 = a; *r1 = e;
}

struct HistArgs {
    const uint8_t* X; const int32_t* xrow; const i64* gh; const uint8_t* nm;
    i64* out; i64 plane; int wide;
    Desc D;
    const uint8_t* fmask;
    const int32_t* fbo;
    int d, B, FGrt, ng, rch, MNb, nnb, maxb, words;
};

template <int FG, bool WORDS>
__device__ __forceinline__ void group_idx(const HistArgs& a, i64 xr, int f0, int fcnt, const int* __restrict__ off_s,
                                          int* idx) {
    if (WORDS) {
        const uint32_t* xw = (const uint32_t*)(a.X + (size_t)xr * a.d);
        if (FG >= 4) {
            uint32_t w[(FG + 3) / 4];
#pragma unroll
            for (int i = 0; i < (FG + 3) / 4; ++i) w[i] = (4 * i < fcnt) ? __ldg(xw + (f0 >> 2) + i) : 0u;
#pragma unroll
            for (int j = 0; j < FG; ++j) idx[j] = off_s[j] + (int)((w[j >> 2] >> (8 * (j & 3))) & 255u);
        } else {
            const uint32_t w = __ldg(xw + (f0 >> 2));
            const int sh0 = 8 * (f0 & 3);
#pragma unroll
            for (int j = 0; j < FG; ++j) idx[j] = off_s[j] + (int)((w >> (sh0 + 8 * j)) & 255u);
        }
    } else {
        const uint8_t* xb = a.X + (size_t)xr * a.d + f0;
#pragma unroll
        for (int j = 0; j < FG; ++j) idx[j] = off_s[j] + ((j < fcnt) ? (int)__ldg(xb + j) : 0);
    }
}

template <int FG, bool WORDS>
__global__ void __launch_bounds__(NM_HIST_MAX_THREADS) hist_kernel(HistArgs a) {
    __shared__ int off_s[NM_MAXFG + 1];
    __shared__ int tbl_s[256];
    __shared__ int dst_s[NM_MAXB];
    __shared__ unsigned char fm_s[NM_MAXFG];
    i64* sh = (i64*)nm_dyn_smem;
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    i64 t = (i64)blockIdx.x;
    const int jb = (int)(t % a.nnb); t /= a.nnb;
    const int g = (int)(t % a.ng); t /= a.ng;
    int i; i64 r0, r1;
    slot_chunk(a.D, t, a.rch, &i, &r0, &r1);
    const int f0 = g * a.FGrt;
    int fcnt = a.d - f0; if (fcnt > a.FGrt) fcnt = a.FGrt;
    const int base = a.fbo[f0];
    const int W = a.fbo[f0 + fcnt] - base;
    const int MNb = a.MNb;
    const int j0 = jb * MNb;
    const size_t frow = (size_t)a.D.slot[i] * a.d;
    const int e0 = (int)a.D.eoff[i];
    int ne = (int)a.D.eoff[i + 1] - e0; if (ne > NM_MAXB) ne = NM_MAXB;
    for (int j = tid; j <= FG; j += bdim) off_s[j] = (j <= fcnt) ? a.fbo[f0 + (j < fcnt ? j : fcnt)] - base : W;
    for (int j = tid; j < FG; j += bdim) fm_s[j] = (j < fcnt) ? a.fmask[frow + f0 + j] : 0;
    for (int k = tid; k < 256; k += bdim) tbl_s[k] = -1;
    const int cells = MNb * W;
    for (int c = tid; c < cells; c += bdim) sh[c] = 0;
    const i64 xb = a.D.xbase[i];
    const i64 xd = xb - a.D.sa[i];
    __syncthreads();
    for (int k = tid; k < ne; k += bdim) { tbl_s[(int)(a.D.node[e0 + k] & 255)] = k; dst_s[k] = (int)a.D.aux[e0 + k]; }
    __syncthreads();
    for (i64 q = r0 + tid; q < r1; q += bdim) {
        const int node = (int)a.nm[q];
        if (node == 255) continue;
        const int l = tbl_s[node] - j0;
        if ((unsigned)l >= (unsigned)MNb) continue;
        const i64 v = a.gh[q];
        if (!v) continue;
        int idx[FG];
        group_idx<FG, WORDS>(a, xb >= 0 ? q + xd : (i64)a.xrow[q], f0, fcnt, off_s, idx);
        i64* tb = sh + (size_t)l * W;
#pragma unroll
        for (int j = 0; j < FG; ++j)
            if (j < fcnt) atom_add_i64_as32_nocarry(tb + idx[j], v);
    }
    __syncthreads();
    const int dB = a.d * a.B;
    for (int c = tid; c < cells; c += bdim) {
        const i64 vv = sh[c];
        if (!vv) continue;
        const int jj = c / W;
        const int idx = c - jj * W;
        const int jf = upper_bound_i32(off_s + 1, fcnt, idx);
        if (!fm_s[jf]) continue;
        const int bin = idx - off_s[jf];
        const int dest = dst_s[j0 + jj];
        const size_t oidx = (size_t)dest * dB + (size_t)(f0 + jf) * a.B + bin;
        if (a.wide) {
            i64 vg, vh; split_word(vv, &vg, &vh);
            if (vg) atom_add_i64(a.out + oidx, vg);
            if (vh) atom_add_i64(a.out + a.plane + oidx, vh);
        } else {
            atom_add_i64(a.out + oidx, vv);
        }
    }
}

struct UpdArgs {
    const uint8_t* X; const int32_t* xrow; uint8_t* nm;
    Desc D;
    const int32_t* sfeat; const int32_t* sthr;
    int32_t* lcount;
    int d, rch;
};

__global__ void update_kernel(UpdArgs a) {
    __shared__ int tbl_s[256];
    __shared__ int f_s[NM_MAXB], t_s[NM_MAXB], c_s[NM_MAXB], o_s[NM_MAXB];
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    int i; i64 r0, r1;
    slot_chunk(a.D, (i64)blockIdx.x, a.rch, &i, &r0, &r1);
    const int e0 = (int)a.D.eoff[i];
    int ne = (int)a.D.eoff[i + 1] - e0; if (ne > NM_MAXB) ne = NM_MAXB;
    for (int k = tid; k < 256; k += bdim) tbl_s[k] = -1;
    __syncthreads();
    for (int k = tid; k < ne; k += bdim) {
        const int o = (int)a.D.aux[e0 + k];
        tbl_s[(int)(a.D.node[e0 + k] & 255)] = k;
        f_s[k] = a.sfeat[o];
        t_s[k] = a.sthr[o];
        o_s[k] = o;
        c_s[k] = 0;
    }
    const i64 xb = a.D.xbase[i];
    const i64 xd = xb - a.D.sa[i];
    __syncthreads();
    const int d = a.d;
    for (i64 q = r0 + tid; q < r1; q += bdim) {
        const int node = (int)a.nm[q];
        if (node == 255) continue;
        const int s = tbl_s[node];
        if (s < 0) continue;
        const i64 xr = xb >= 0 ? q + xd : (i64)a.xrow[q];
        const int x = (int)__ldg(a.X + (size_t)xr * d + f_s[s]);
        const int right = x > t_s[s] ? 1 : 0;
        a.nm[q] = (uint8_t)(2 * node + 1 + right);
        if (!right) atomicAdd(&c_s[s], 1);
    }
    __syncthreads();
    for (int k = tid; k < ne; k += bdim)
        if (c_s[k]) atomicAdd(a.lcount + o_s[k], c_s[k]);
}

struct LeafArgs {
    float* pred; const uint8_t* nm;
    Desc D;
    const float* lv;
    int* err;
    int rch;
};

__global__ void leaf_kernel(LeafArgs a) {
    __shared__ float v_s[256];
    __shared__ unsigned char ok_s[256];
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    int i; i64 r0, r1;
    slot_chunk(a.D, (i64)blockIdx.x, a.rch, &i, &r0, &r1);
    const int e0 = (int)a.D.eoff[i];
    int ne = (int)a.D.eoff[i + 1] - e0; if (ne > NM_MAXB) ne = NM_MAXB;
    for (int k = tid; k < 256; k += bdim) ok_s[k] = 0;
    __syncthreads();
    for (int k = tid; k < ne; k += bdim) {
        const int n = (int)(a.D.node[e0 + k] & 255);
        v_s[n] = a.lv[a.D.aux[e0 + k]];
        ok_s[n] = 1;
    }
    __syncthreads();
    int bad = 0;
    for (i64 q = r0 + tid; q < r1; q += bdim) {
        const int node = (int)a.nm[q];
        if (node == 255) continue;
        if (!ok_s[node]) { ++bad; continue; }
        a.pred[q] = a.pred[q] + v_s[node];
    }
    if (bad) atomicAdd(a.err, bad);
}

struct CmpArgs {
    const uint8_t* nm;
    Desc D;
    int32_t* ccnt;
    i64* cbase;
    int32_t* wl;
    int* err;
    int rch, maxb;
};

__global__ void compact_count_kernel(CmpArgs a) {
    __shared__ int tbl_s[256];
    __shared__ int c_s[NM_MAXB];
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    const i64 ch = (i64)blockIdx.x;
    int i; i64 r0, r1;
    slot_chunk(a.D, ch, a.rch, &i, &r0, &r1);
    const int e0 = (int)a.D.eoff[i];
    int ne = (int)a.D.eoff[i + 1] - e0; if (ne > a.maxb) ne = a.maxb;
    for (int k = tid; k < 256; k += bdim) tbl_s[k] = -1;
    for (int k = tid; k < a.maxb; k += bdim) c_s[k] = 0;
    __syncthreads();
    for (int k = tid; k < ne; k += bdim) tbl_s[(int)(a.D.node[e0 + k] & 255)] = k;
    __syncthreads();
    for (i64 q = r0 + tid; q < r1; q += bdim) {
        const int node = (int)a.nm[q];
        if (node == 255) continue;
        const int j = tbl_s[node];
        if (j >= 0) atomicAdd(&c_s[j], 1);
    }
    __syncthreads();
    for (int k = tid; k < a.maxb; k += bdim) a.ccnt[ch * a.maxb + k] = c_s[k];
}

__global__ void compact_offsets_kernel(CmpArgs a) {
    const i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (i64)a.D.ns * a.maxb) return;
    const int i = (int)(t / a.maxb), j = (int)(t - (i64)i * a.maxb);
    const int e0 = (int)a.D.eoff[i], ne = (int)a.D.eoff[i + 1] - e0;
    if (j >= ne) return;
    const i64 beg = a.D.aux[e0 + j];
    i64 run = beg;
    const i64 c0 = i ? a.D.cum_ch[i - 1] : 0, c1 = a.D.cum_ch[i];
    for (i64 ch = c0; ch < c1; ++ch) {
        a.cbase[ch * a.maxb + j] = run;
        run += a.ccnt[ch * a.maxb + j];
    }
    if (run - beg != a.D.aux2[e0 + j]) atomicAdd(a.err, 1);
}

__global__ void compact_scatter_kernel(CmpArgs a) {
    __shared__ int tbl_s[256];
    __shared__ i64 base_s[NM_MAXB];
    __shared__ i64 end_s[NM_MAXB];
    __shared__ int tot_s[NM_MAXB];
    __shared__ int wc_s[32 * NM_MAXB];
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    const int lane = tid & 31, w = tid >> 5, nw = bdim >> 5;
    const i64 ch = (i64)blockIdx.x;
    int i; i64 r0, r1;
    slot_chunk(a.D, ch, a.rch, &i, &r0, &r1);
    const int e0 = (int)a.D.eoff[i];
    int ne = (int)a.D.eoff[i + 1] - e0; if (ne > a.maxb) ne = a.maxb;
    for (int k = tid; k < 256; k += bdim) tbl_s[k] = -1;
    for (int k = tid; k < a.maxb; k += bdim) { base_s[k] = a.cbase[ch * a.maxb + k]; end_s[k] = 0; }
    __syncthreads();
    for (int k = tid; k < ne; k += bdim) {
        tbl_s[(int)(a.D.node[e0 + k] & 255)] = k;
        end_s[k] = a.D.aux[e0 + k] + a.D.aux2[e0 + k];
    }
    __syncthreads();
    const unsigned lt_mask = (1u << lane) - 1u;
    int bad = 0;
    for (i64 t0 = r0; t0 < r1; t0 += bdim) {
        for (int k = tid; k < nw * a.maxb; k += bdim) wc_s[k] = 0;
        __syncthreads();
        const i64 q = t0 + tid;
        int j = -1;
        if (q < r1) {
            const int node = (int)a.nm[q];
            if (node != 255) j = tbl_s[node];
        }
        const unsigned peers = __match_any_sync(0xFFFFFFFFu, j);
        const int rank = __popc(peers & lt_mask);
        if (j >= 0 && rank == 0) wc_s[w * a.maxb + j] = __popc(peers);
        __syncthreads();
        for (int k = tid; k < a.maxb; k += bdim) {
            int run = 0;
            for (int ww = 0; ww < nw; ++ww) { const int c = wc_s[ww * a.maxb + k]; wc_s[ww * a.maxb + k] = run; run += c; }
            tot_s[k] = run;
        }
        __syncthreads();
        if (j >= 0) {
            const i64 p = base_s[j] + wc_s[w * a.maxb + j] + rank;
            if (p < end_s[j]) a.wl[p] = (int32_t)q; else ++bad;
        }
        __syncthreads();
        for (int k = tid; k < a.maxb; k += bdim) base_s[k] += tot_s[k];
        __syncthreads();
    }
    if (bad) atomicAdd(a.err, bad);
}

struct FinArgs {
    const float* og; const int32_t* of; const int32_t* ob; const i64* ogl; const i64* ohl;
    int32_t* RB;
    int C, NFB;
    i64 C_chunk, pos0;
};

__global__ void split_finish_kernel(FinArgs a) {
    const int c = (int)(blockIdx.x * (i64)blockDim.x + threadIdx.x);
    if (c >= a.C) return;
    const float* row = a.og + (size_t)c * a.NFB;
    float best = row[0];
    int j = 0;
    bool nan = isnan(best);
    for (int b = 1; b < a.NFB; ++b) {
        const float v = row[b];
        if (isnan(v)) nan = true;
        else if (!nan && v > best) { best = v; j = b; }
    }
    if (nan) j = 0;
    const size_t o = (size_t)c * a.NFB + j;
    const i64 p = a.pos0 + c;
    i64* GL = (i64*)a.RB;
    i64* HL = (i64*)(a.RB + 2 * a.C_chunk);
    float* bg = (float*)(a.RB + 4 * a.C_chunk);
    int32_t* bf = a.RB + 5 * a.C_chunk;
    int32_t* bb = a.RB + 6 * a.C_chunk;
    GL[p] = a.ogl[o];
    HL[p] = a.ohl[o];
    bg[p] = a.og[o];
    bf[p] = a.of[o];
    bb[p] = a.ob[o];
}

struct ScalesArgs {
    const float* gmax; const float* hmax; const int8_t* bits;
    float* sg32; float* sh32; double* sg28; double* sh28; uint8_t* is28; double* sg; double* sh;
    int S, has_hess;
};

__global__ void scales_kernel(ScalesArgs a) {
    for (int s = (int)(blockIdx.x * (i64)blockDim.x + threadIdx.x); s < a.S; s += (int)(gridDim.x * (i64)blockDim.x)) {
        const float g = a.gmax[s];
        const float sg32 = g > 0.0f ? 32767.0f / g : 1.0f;
        const double g64 = (double)g;
        const double sg28 = g64 > 0.0 ? (1.0 / g64) * 134217727.0 : 1.0;
        const bool is28 = a.bits[s] == 28;
        float sh32 = 1.0f;
        double sh28 = 1.0, sh = 1.0;
        if (a.has_hess) {
            const float h = a.hmax[s];
            sh32 = h > 0.0f ? 65535.0f / h : 1.0f;
            const double h64 = (double)h;
            sh28 = h64 > 0.0 ? (1.0 / h64) * 268435455.0 : 1.0;
            sh = is28 ? sh28 : (double)sh32;
        }
        a.sg32[s] = sg32; a.sg28[s] = sg28; a.is28[s] = is28 ? 1 : 0;
        a.sg[s] = is28 ? sg28 : (double)sg32;
        a.sh32[s] = sh32; a.sh28[s] = sh28; a.sh[s] = sh;
    }
}

struct RowsumArgs {
    const i64* gh; const uint8_t* work; uint8_t* nm;
    Desc D;
    i64* G; i64* H; int32_t* cnt;
    int rch;
};

__global__ void rowsum_kernel(RowsumArgs a) {
    __shared__ i64 sG[32], sH[32];
    __shared__ int sC[32];
    const int tid = (int)threadIdx.x, bdim = (int)blockDim.x;
    const int lane = tid & 31, w = tid >> 5, nw = bdim >> 5;
    int i; i64 r0, r1;
    slot_chunk(a.D, (i64)blockIdx.x, a.rch, &i, &r0, &r1);
    i64 g = 0, h = 0;
    int c = 0;
    for (i64 q = r0 + tid; q < r1; q += bdim) {
        const bool wq = a.work[q] != 0;
        a.nm[q] = wq ? (uint8_t)0 : (uint8_t)255;
        if (wq) {
            i64 vg, vh; split_word(a.gh[q], &vg, &vh);
            g += vg; h += vh; ++c;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        g += __shfl_down_sync(0xFFFFFFFFu, g, o);
        h += __shfl_down_sync(0xFFFFFFFFu, h, o);
        c += __shfl_down_sync(0xFFFFFFFFu, c, o);
    }
    if (lane == 0) { sG[w] = g; sH[w] = h; sC[w] = c; }
    __syncthreads();
    if (tid == 0) {
        i64 tg = 0, th = 0;
        int tc = 0;
        for (int k = 0; k < nw; ++k) { tg += sG[k]; th += sH[k]; tc += sC[k]; }
        if (tc) {
            const int s = (int)a.D.slot[i];
            atom_add_i64(a.G + s, tg);
            atom_add_i64(a.H + s, th);
            atomicAdd(a.cnt + s, tc);
        }
    }
}

struct GradArgs {
    const float* pred; const void* y;
    const i64* mrow_base; const i64* mrow_len;
    float* g; float* h;
    i64 n, Nm;
    int K;
};

__device__ __forceinline__ float clamp_min_f32(float v, float lo) { return isnan(v) ? v : fmaxf(v, lo); }
__device__ __forceinline__ double torch_maximum_f64(double a, double b) {
    return (a != a) ? a : ((b != b) ? b : fmax(a, b));
}

__global__ void grad_logistic_kernel(GradArgs a) {
    const float* y = (const float*)a.y;
    const float lo = (float)1e-16;
    for (i64 q = blockIdx.x * (i64)blockDim.x + threadIdx.x; q < a.n; q += (i64)gridDim.x * blockDim.x) {
        const double x = (double)a.pred[q];
        const double e = exp(-x);
        const double t = 1.0 + e;
        const double p64 = 1.0 / t;
        const float p = (float)p64;
        const float gq = p - y[q];
        const float v = p * (1.0f - p);
        a.g[q] = gq;
        a.h[q] = clamp_min_f32(v, lo);
    }
}

template <int KMAX>
__global__ void grad_softmax_kernel(GradArgs a) {
    const i64* y = (const i64*)a.y;
    const int K = a.K;
    const float lo = (float)1e-16;
    for (i64 r = blockIdx.x * (i64)blockDim.x + threadIdx.x; r < a.Nm; r += (i64)gridDim.x * blockDim.x) {
        const i64 b = a.mrow_base[r], L = a.mrow_len[r];
        double z[KMAX];
#pragma unroll
        for (int k = 0; k < KMAX; ++k) if (k < K) z[k] = (double)a.pred[b + (i64)k * L];
        double mx = z[0];
#pragma unroll
        for (int k = 1; k < KMAX; ++k) if (k < K) mx = torch_maximum_f64(mx, z[k]);
#pragma unroll
        for (int k = 0; k < KMAX; ++k) if (k < K) z[k] = exp(z[k] - mx);
        double s = z[0];
#pragma unroll
        for (int k = 1; k < KMAX; ++k) if (k < K) s = s + z[k];
        const i64 yk = y[b];
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            if (k < K) {
                const float P = (float)(z[k] / s);
                const float v = (2.0f * P) * (1.0f - P);
                const i64 q = b + (i64)k * L;
                a.g[q] = P - ((i64)k == yk ? 1.0f : 0.0f);
                a.h[q] = clamp_min_f32(v, lo);
            }
        }
    }
}

struct RowmaskArgs {
    const uint8_t* is_train; const int32_t* crow; const int32_t* slot;
    const i64* seed; const i64* lr; const double* sub; const i64* flags;
    uint8_t* work; double* u_out;
    i64 n; int S;
};

__device__ __forceinline__ uint32_t mix32_u(uint32_t x) {
    x = (x ^ (x >> 16)) * 0x7FEB352Du;
    x = (x ^ (x >> 15)) * 0x846CA68Bu;
    return x ^ (x >> 16);
}
__device__ __forceinline__ uint32_t hash_step(uint32_t h, i64 key) {
    return mix32_u(h * 0x9E3779B1u + mix32_u((uint32_t)(u64)key * 0x85EBCA77u + 0xC2B2AE3Du));
}

__global__ void rowmask_kernel(RowmaskArgs a) {
    for (i64 q = blockIdx.x * (i64)blockDim.x + threadIdx.x; q < a.n; q += (i64)gridDim.x * blockDim.x) {
        const int s = a.slot[q];
        if (s < 0 || s >= a.S) {
            a.work[q] = 0;
            if (a.u_out) a.u_out[q] = -1.0;
            continue;
        }
        const i64 fl = a.flags[s];
        bool w = (a.is_train[q] != 0) && ((fl & 1) != 0);
        if (a.u_out != nullptr || (w && (fl & 2) != 0)) {
            uint32_t h = hash_step(0u, 1);
            h = hash_step(h, a.seed[s]);
            h = hash_step(h, a.lr[s]);
            h = hash_step(h, (i64)a.crow[q]);
            const double u = (double)h / 4294967296.0;
            if (a.u_out) a.u_out[q] = u;
            if (w && (fl & 2) != 0) w = u < a.sub[s];
        }
        a.work[q] = w ? (uint8_t)1 : (uint8_t)0;
    }
}

static int set_smem(const void* fn, int smem_bytes) {
    cudaFuncAttributes attrs;
    cudaError_t e = cudaFuncGetAttributes(&attrs, fn);
    if (e != cudaSuccess) return (int)e;
    if (smem_bytes > attrs.maxDynamicSharedSizeBytes) {
        e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (e != cudaSuccess) return (int)e;
    }
    return 0;
}

template <int FG>
static int hist_go(const HistArgs& a, i64 grid, int threads, int smem, int auto_threads, cudaStream_t st) {
    const void* fn = a.words ? (const void*)hist_kernel<FG, true> : (const void*)hist_kernel<FG, false>;
    int rc = set_smem(fn, smem);
    if (rc) return rc;
    if (auto_threads && threads > 256) {
        int nb_hi = 0, nb_lo = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb_hi, fn, threads, smem) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb_lo, fn, 256, smem) == cudaSuccess &&
            nb_lo * 256 > nb_hi * threads)
            threads = 256;
        (void)cudaGetLastError();
    }
    if (a.words) hist_kernel<FG, true><<<(unsigned)grid, threads, smem, st>>>(a);
    else hist_kernel<FG, false><<<(unsigned)grid, threads, smem, st>>>(a);
    return (int)cudaGetLastError();
}

static int check_threads(int threads, int maxt) {
    return (threads < 32 || threads > maxt || (threads & 31)) ? -3 : 0;
}

}

using namespace mgnm;

NM_EXPORT int mgnm_version() { return MGNM_VERSION; }

NM_EXPORT int mgnm_hist_occupancy(int FG, int words, int threads, int smem_bytes, int* blocks_out, int* regs_out) {
    const void* fn = nullptr;
    switch (FG) {
        case 1: fn = words ? (const void*)hist_kernel<1, true> : (const void*)hist_kernel<1, false>; break;
        case 2: fn = words ? (const void*)hist_kernel<2, true> : (const void*)hist_kernel<2, false>; break;
        case 4: fn = words ? (const void*)hist_kernel<4, true> : (const void*)hist_kernel<4, false>; break;
        case 8: fn = words ? (const void*)hist_kernel<8, true> : (const void*)hist_kernel<8, false>; break;
        case 16: fn = words ? (const void*)hist_kernel<16, true> : (const void*)hist_kernel<16, false>; break;
        case 28: fn = words ? (const void*)hist_kernel<28, true> : (const void*)hist_kernel<28, false>; break;
        case 32: fn = words ? (const void*)hist_kernel<32, true> : (const void*)hist_kernel<32, false>; break;
        default: return -10;
    }
    int rc = set_smem(fn, smem_bytes);
    if (rc) return rc;
    cudaFuncAttributes at;
    cudaError_t e = cudaFuncGetAttributes(&at, fn);
    if (e != cudaSuccess) return (int)e;
    *regs_out = at.numRegs;
    return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_out, fn, threads, smem_bytes);
}

NM_EXPORT int mgnm_hist(const void* X, const void* xrow, const void* gh, const void* nm, void* out, i64 plane, int wide,
                        const void* desc, int ns, int E, const void* fmask, const void* fbo, int d, int B, int FG,
                        int rch, int MNb, int nnb, int maxb, int words, i64 n_chunks, void* stream, int threads,
                        int smem_bytes, int auto_threads) {
    HistArgs a;
    a.X = (const uint8_t*)X; a.xrow = (const int32_t*)xrow; a.gh = (const i64*)gh; a.nm = (const uint8_t*)nm;
    a.out = (i64*)out; a.plane = plane; a.wide = wide;
    a.D = make_desc(desc, ns, E);
    a.fmask = (const uint8_t*)fmask; a.fbo = (const int32_t*)fbo;
    a.d = d; a.B = B; a.FGrt = FG; a.rch = rch; a.MNb = MNb; a.nnb = nnb; a.maxb = maxb; a.words = words;
    if (rch > 65536 || rch < 1) return -5;
    if (maxb > NM_MAXB || MNb < 1 || nnb < 1) return -6;
    if (words && (d % 4)) return -8;
    if (FG < 1 || FG > NM_MAXFG) return -9;
    if (FG < 4 && (4 % FG)) return -9;
    if (FG >= 4 && (FG % 4) && words) return -9;
    a.ng = (d + FG - 1) / FG;
    const i64 grid = n_chunks * (i64)a.ng * nnb;
    if (grid <= 0 || ns <= 0) return 0;
    if (grid > 2147483647LL) return -2;
    if (check_threads(threads, NM_HIST_MAX_THREADS)) return -3;
    cudaStream_t st = (cudaStream_t)stream;
    switch (FG) {
        case 1: return hist_go<1>(a, grid, threads, smem_bytes, auto_threads, st);
        case 2: return hist_go<2>(a, grid, threads, smem_bytes, auto_threads, st);
        case 4: return hist_go<4>(a, grid, threads, smem_bytes, auto_threads, st);
        case 8: return hist_go<8>(a, grid, threads, smem_bytes, auto_threads, st);
        case 16: return hist_go<16>(a, grid, threads, smem_bytes, auto_threads, st);
        case 28: return hist_go<28>(a, grid, threads, smem_bytes, auto_threads, st);
        case 32: return hist_go<32>(a, grid, threads, smem_bytes, auto_threads, st);
        default: return -10;
    }
}

NM_EXPORT int mgnm_update(const void* X, const void* xrow, void* nm, const void* desc, int ns, int E, const void* sfeat,
                          const void* sthr, void* lcount, int d, int rch, int maxs, i64 n_chunks, void* stream,
                          int threads) {
    UpdArgs a;
    a.X = (const uint8_t*)X; a.xrow = (const int32_t*)xrow; a.nm = (uint8_t*)nm;
    a.D = make_desc(desc, ns, E);
    a.sfeat = (const int32_t*)sfeat; a.sthr = (const int32_t*)sthr; a.lcount = (int32_t*)lcount;
    a.d = d; a.rch = rch;
    if (maxs > NM_MAXB || maxs < 1) return -6;
    if (E <= 0 || ns <= 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    cudaError_t e = cudaMemsetAsync(lcount, 0, (size_t)E * sizeof(int32_t), st);
    if (e != cudaSuccess) return (int)e;
    if (n_chunks <= 0) return 0;
    if (n_chunks > 2147483647LL) return -2;
    if (check_threads(threads, 1024)) return -3;
    update_kernel<<<(unsigned)n_chunks, threads, 0, st>>>(a);
    return (int)cudaGetLastError();
}

NM_EXPORT int mgnm_leaf(void* pred, const void* nm, const void* desc, int ns, int E, const void* lv, void* err, int rch,
                        i64 n_chunks, void* stream, int threads) {
    LeafArgs a;
    a.pred = (float*)pred; a.nm = (const uint8_t*)nm;
    a.D = make_desc(desc, ns, E);
    a.lv = (const float*)lv; a.err = (int*)err; a.rch = rch;
    if (n_chunks <= 0 || ns <= 0) return 0;
    if (n_chunks > 2147483647LL) return -2;
    if (check_threads(threads, 1024)) return -3;
    leaf_kernel<<<(unsigned)n_chunks, threads, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}

NM_EXPORT int mgnm_compact(const void* nm, const void* desc, int ns, int E, void* ccnt, void* cbase, void* wl, void* err,
                           int rch, int maxb, i64 n_chunks, void* stream, int threads) {
    CmpArgs a;
    a.nm = (const uint8_t*)nm;
    a.D = make_desc(desc, ns, E);
    a.ccnt = (int32_t*)ccnt; a.cbase = (i64*)cbase; a.wl = (int32_t*)wl; a.err = (int*)err;
    a.rch = rch; a.maxb = maxb;
    if (maxb > NM_MAXB || maxb < 1) return -6;
    if (n_chunks <= 0 || ns <= 0 || E <= 0) return 0;
    if (n_chunks > 2147483647LL) return -2;
    if (check_threads(threads, 1024)) return -3;
    if ((threads >> 5) * maxb > 32 * NM_MAXB) return -7;
    cudaStream_t st = (cudaStream_t)stream;
    compact_count_kernel<<<(unsigned)n_chunks, threads, 0, st>>>(a);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    const i64 nt = (i64)ns * maxb;
    compact_offsets_kernel<<<(unsigned)((nt + 255) / 256), 256, 0, st>>>(a);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    compact_scatter_kernel<<<(unsigned)n_chunks, threads, 0, st>>>(a);
    return (int)cudaGetLastError();
}

NM_EXPORT int mgnm_split_finish(const void* og, const void* of, const void* ob, const void* ogl, const void* ohl, int C,
                                int NFB, void* RB, i64 C_chunk, i64 pos0, void* stream, int threads) {
    FinArgs a;
    a.og = (const float*)og; a.of = (const int32_t*)of; a.ob = (const int32_t*)ob;
    a.ogl = (const i64*)ogl; a.ohl = (const i64*)ohl;
    a.RB = (int32_t*)RB; a.C = C; a.NFB = NFB; a.C_chunk = C_chunk; a.pos0 = pos0;
    if (C <= 0) return 0;
    if (NFB < 1) return -11;
    if (C_chunk < 1 || pos0 < 0 || pos0 + (i64)C > C_chunk) return -12;
    if (((uintptr_t)RB) & 7) return -13;
    if (check_threads(threads, 1024)) return -3;
    const i64 grid = ((i64)C + threads - 1) / threads;
    if (grid > 2147483647LL) return -2;
    split_finish_kernel<<<(unsigned)grid, threads, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}

NM_EXPORT int mgnm_scales(const void* gmax, const void* hmax, const void* bits, int S, int has_hess, void* sg32,
                          void* sh32, void* sg28, void* sh28, void* is28, void* sg, void* sh, void* stream,
                          int threads) {
    ScalesArgs a;
    a.gmax = (const float*)gmax; a.hmax = (const float*)hmax; a.bits = (const int8_t*)bits;
    a.sg32 = (float*)sg32; a.sh32 = (float*)sh32; a.sg28 = (double*)sg28; a.sh28 = (double*)sh28;
    a.is28 = (uint8_t*)is28; a.sg = (double*)sg; a.sh = (double*)sh;
    a.S = S; a.has_hess = has_hess ? 1 : 0;
    if (S <= 0) return 0;
    if (!gmax || !bits || !sg32 || !sh32 || !sg28 || !sh28 || !is28 || !sg || !sh || (has_hess && !hmax)) return -15;
    if (check_threads(threads, 1024)) return -3;
    const i64 grid = ((i64)S + threads - 1) / threads;
    scales_kernel<<<(unsigned)(grid > 65535 ? 65535 : grid), threads, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}

NM_EXPORT int mgnm_rowsum(const void* gh, const void* work, void* nm, const void* desc, int ns, void* G, void* H,
                          void* cnt, int rch, i64 n_chunks, void* stream, int threads) {
    RowsumArgs a;
    a.gh = (const i64*)gh; a.work = (const uint8_t*)work; a.nm = (uint8_t*)nm;
    a.D = make_desc(desc, ns, 0);
    a.G = (i64*)G; a.H = (i64*)H; a.cnt = (int32_t*)cnt; a.rch = rch;
    if (ns <= 0 || n_chunks <= 0) return 0;
    if (!gh || !work || !nm || !desc || !G || !H || !cnt) return -15;
    if (rch < 1) return -5;
    if (n_chunks > 2147483647LL) return -2;
    if (check_threads(threads, 1024)) return -3;
    rowsum_kernel<<<(unsigned)n_chunks, threads, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}

NM_EXPORT int mgnm_zero(void* ptr, i64 nbytes, void* stream) {
    if (nbytes < 0) return -14;
    if (nbytes == 0) return 0;
    if (!ptr) return -15;
    return (int)cudaMemsetAsync(ptr, 0, (size_t)nbytes, (cudaStream_t)stream);
}

NM_EXPORT int mgnm_grad(int loss, int K, const void* pred, const void* y, const void* mrow_base, const void* mrow_len,
                        void* g, void* h, i64 n, i64 Nm, void* stream, int threads) {
    GradArgs a;
    a.pred = (const float*)pred; a.y = y; a.mrow_base = (const i64*)mrow_base; a.mrow_len = (const i64*)mrow_len;
    a.g = (float*)g; a.h = (float*)h; a.n = n; a.Nm = Nm; a.K = K;
    if (loss != 0 && loss != 1) return -18;
    if (n < 0 || Nm < 0) return -17;
    if (n == 0) return 0;
    if (!pred || !y || !g || !h) return -15;
    if (check_threads(threads, 1024)) return -3;
    cudaStream_t st = (cudaStream_t)stream;
    if (loss == 0) {
        i64 grid = (n + threads - 1) / threads;
        if (grid > NM_GRAD_MAX_GRID) grid = NM_GRAD_MAX_GRID;
        grad_logistic_kernel<<<(unsigned)grid, threads, 0, st>>>(a);
        return (int)cudaGetLastError();
    }
    if (K < 1 || K > NM_MAXK) return -16;
    if (Nm * (i64)K != n) return -17;
    if (!mrow_base || !mrow_len) return -15;
    i64 grid = (Nm + threads - 1) / threads;
    if (grid > NM_GRAD_MAX_GRID) grid = NM_GRAD_MAX_GRID;
    if (K <= 4) grad_softmax_kernel<4><<<(unsigned)grid, threads, 0, st>>>(a);
    else if (K <= 8) grad_softmax_kernel<8><<<(unsigned)grid, threads, 0, st>>>(a);
    else if (K <= 16) grad_softmax_kernel<16><<<(unsigned)grid, threads, 0, st>>>(a);
    else grad_softmax_kernel<32><<<(unsigned)grid, threads, 0, st>>>(a);
    return (int)cudaGetLastError();
}

NM_EXPORT int mgnm_rowmask(const void* is_train, const void* crow, const void* slot, const void* tab, int S, void* work,
                           void* u_out, i64 n, void* stream, int threads) {
    RowmaskArgs a;
    a.is_train = (const uint8_t*)is_train; a.crow = (const int32_t*)crow; a.slot = (const int32_t*)slot;
    const i64* t = (const i64*)tab;
    a.seed = t; a.lr = t + S; a.sub = (const double*)(t + 2 * (size_t)S); a.flags = t + 3 * (size_t)S;
    a.work = (uint8_t*)work; a.u_out = (double*)u_out; a.n = n; a.S = S;
    if (n < 0 || S < 1) return -17;
    if (n == 0) return 0;
    if (!is_train || !crow || !slot || !tab || !work) return -15;
    if (check_threads(threads, 1024)) return -3;
    i64 grid = (n + threads - 1) / threads;
    if (grid > NM_GRAD_MAX_GRID) grid = NM_GRAD_MAX_GRID;
    rowmask_kernel<<<(unsigned)grid, threads, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
}
