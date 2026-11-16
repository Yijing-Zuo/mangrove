import os
import sys
import ctypes
import math
import weakref
from pathlib import Path

import numpy as np
import torch

from . import kernels_ref as KR
from .kernels_ref import (G16, H16, G28, H28, KRTEPS, NEG_INF, mix32, hash_u01, grad_hess, split_gh,
                          leaf_values, metric_fixed_point, _expand_segments)

try:
    from .kernels_ref import ROW_STREAM, COL_STREAM
except ImportError:
    pass

EMU = os.environ.get("MANGROVE_EMU", "0") == "1"
if EMU:
    os.environ.setdefault("TRITON_INTERPRET", "1")
DEBUG_CHECKS = os.environ.get("MANGROVE_DEBUG_CHECKS", "0") == "1"

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception as _e:
    HAVE_TRITON = False
    _TRITON_ERR = _e

BH_RCH = int(os.environ.get("MANGROVE_BH_RCH", "4096"))
BH_THREADS = int(os.environ.get("MANGROVE_BH_THREADS", "256"))
BH_SMEM_KB = int(os.environ.get("MANGROVE_BH_SMEM_KB", "48"))
BH_SHARED_FACTOR = float(os.environ.get("MANGROVE_BH_SHARED_FACTOR", "2.0"))
SEG_RCH = int(os.environ.get("MANGROVE_SEG_RCH", "4096"))
THREADS = int(os.environ.get("MANGROVE_THREADS", "256"))
EMU_THREADS = int(os.environ.get("MANGROVE_EMU_THREADS", "64"))
MAX_GROUP = 512
EXPECTED_LIB_VERSION = 4
BUILD_HIST_NARROW = True

CUDA_DIR = Path(__file__).resolve().parent.parent / "cuda"
_LIB = {}


def _lib(dev):
    key = "emu" if dev.type == "cpu" else "cuda"
    lib = _LIB.get(key)
    if lib is not None:
        return lib
    if key == "emu":
        if not EMU:
            raise RuntimeError("kernels_gpu called with CPU tensors; set MANGROVE_EMU=1 to use the CPU emulation build")
        path = CUDA_DIR / "libmangrove_emu.so"
        hint = "build it with: bash cuda/build.sh emu"
    else:
        path = CUDA_DIR / ("mangrove.dll" if sys.platform.startswith("win") else "libmangrove.so")
        hint = ("build it from cuda/mg_kernels.cu and cuda/mg_nodemap.cu with nvcc -O3 -shared -fmad=false" if sys.platform.startswith("win")
                else "build it with: bash cuda/build.sh")
    if not path.exists():
        raise RuntimeError(f"Mangrove kernel library not found: {path}\n  -> {hint}")
    lib = ctypes.CDLL(str(path))
    P, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    lib.mg_version.restype = I
    lib.mg_version.argtypes = []
    v = lib.mg_version()
    if v != EXPECTED_LIB_VERSION:
        raise RuntimeError(f"{path} has version {v}, kernels_gpu.py expects {EXPECTED_LIB_VERSION}: rebuild ({hint})")
    lib.mg_build_hist.restype = I
    lib.mg_build_hist.argtypes = [P] * 9 + [L] + [P] * 4 + [I] * 6 + [L, I, L, P, I, I, I]
    lib.mg_node_sums.restype = I
    lib.mg_node_sums.argtypes = [P] * 5 + [I, I, L, P, P, P, I]
    lib.mg_leaf_update.restype = I
    lib.mg_leaf_update.argtypes = [P] * 6 + [I, I, L, P, I]
    lib.mg_partition.restype = I
    lib.mg_partition.argtypes = [P] * 9 + [I, I, I, L, P, P, P, P, I, P]
    lib.mg_route_update.restype = I
    lib.mg_route_update.argtypes = [P] * 5 + [L] + [P] * 4 + [I] + [P, P, L, I, I, P, P, I]
    _LIB[key] = lib
    return lib


def _stream(dev):
    if dev.type == "cuda":
        return torch.cuda.current_stream(dev).cuda_stream
    return 0


def _threads(dev, n):
    return EMU_THREADS if dev.type == "cpu" else n


def _check(rc, name):
    if rc != 0:
        raise RuntimeError(f"{name} failed with code {rc} (negative = argument check in mg_kernels.cu, positive = cudaError_t)")


def _c(t, dtype=None):
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t.contiguous()


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


_CONST = {}


def _dconst(value, dtype, dev):
    key = (str(dev), dtype, float(value))
    t = _CONST.get(key)
    if t is None:
        t = _CONST[key] = torch.tensor(value, dtype=dtype, device=dev)
    return t


def _h2d_async(a, dev):
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dev.type == "cuda":
        return t.pin_memory().to(dev, non_blocking=True)
    return t


def _require_triton():
    if not HAVE_TRITON:
        raise RuntimeError(f"triton is required by kernels_gpu (quantize/split_search/subtract_into/decode_hist): {_TRITON_ERR}")


class _XMeta:
    def __init__(self, X):
        self.nbins = (X.amax(0).to(torch.int64) + 1).tolist()
        d = len(self.nbins)
        off = [0]
        for nb in self.nbins:
            off.append(off[-1] + nb)
        self.total_cells = off[-1]
        self.fbo = torch.tensor(off, dtype=torch.int32, device=X.device)
        self.mean_nbins = self.total_cells / max(1, d)
        self._groups = {}

    def groups(self, smem_cells):
        g = self._groups.get(smem_cells)
        if g is None:
            starts, acc, cnt = [0], 0, 0
            for f, nb in enumerate(self.nbins):
                if f > 0 and (acc + nb > smem_cells or cnt >= MAX_GROUP):
                    starts.append(f)
                    acc, cnt = 0, 0
                acc += nb
                cnt += 1
            starts.append(len(self.nbins))
            g = (torch.tensor(starts, dtype=torch.int32, device=self.fbo.device), len(starts) - 1)
            self._groups[smem_cells] = g
        return g


_XMETA = {}


def _xmeta(X):
    e = _XMETA.get(id(X))
    if e is not None and e[0]() is X and e[1] == X._version and e[2] == X.data_ptr():
        return e[3]
    if len(_XMETA) > 64:
        for k in [k for k, v in _XMETA.items() if v[0]() is None]:
            del _XMETA[k]
    m = _XMeta(X)
    _XMETA[id(X)] = (weakref.ref(X), X._version, X.data_ptr(), m)
    return m


def _chunks(beg, end, rch):
    ln = (end - beg).to(torch.int64)
    cum = torch.cumsum((ln + (rch - 1)) // rch, 0)
    return cum, int(cum[-1].item())


def build_hist_sizes(X_all, ln):
    meta = _xmeta(X_all)
    thr = max(1, int(math.ceil(BH_SHARED_FACTOR * meta.mean_nbins)))
    sh = ln >= thr
    return int(ln[~sh].sum()), int(((ln[sh] + (BH_RCH - 1)) // BH_RCH).sum())


def partition_sizes(ln):
    return int(((ln + (SEG_RCH - 1)) // SEG_RCH).sum()), int(ln.sum())


def build_hist_grid(X_all, ln):
    meta = _xmeta(X_all)
    thr = max(1, int(math.ceil(BH_SHARED_FACTOR * meta.mean_nbins)))
    ln = np.asarray(ln, dtype=np.int64)
    sh = ln >= thr
    cum_tiny = np.cumsum(np.where(sh, 0, ln)).astype(np.int64, copy=False)
    cum_ch = np.cumsum(np.where(sh, (ln + (BH_RCH - 1)) // BH_RCH, 0)).astype(np.int64, copy=False)
    sizes = (int(cum_tiny[-1]), int(cum_ch[-1])) if len(ln) else (0, 0)
    return sizes, cum_tiny, cum_ch


def partition_grid(ln):
    ln = np.asarray(ln, dtype=np.int64)
    cum_ch = np.cumsum((ln + (SEG_RCH - 1)) // SEG_RCH).astype(np.int64, copy=False)
    cum_len = np.cumsum(ln).astype(np.int64, copy=False)
    sizes = (int(cum_ch[-1]), int(cum_len[-1])) if len(ln) else (0, 0)
    return sizes, cum_ch, cum_len


def segment_grid(ln, rch=None):
    rch = SEG_RCH if rch is None else rch
    ln = np.asarray(ln, dtype=np.int64)
    cum = np.cumsum((ln + (rch - 1)) // rch).astype(np.int64, copy=False)
    return cum, (int(cum[-1]) if len(ln) else 0)


def build_hist(X_all, xrow, gh, wl, beg, end, dest, fmask, out, wide, narrow=None, sizes=None, cums=None):
    nb = len(beg)
    if nb == 0:
        return
    dev = X_all.device
    lib = _lib(dev)
    d = X_all.shape[1]
    B = out.shape[-1]
    assert X_all.dtype == torch.uint8 and X_all.is_contiguous()
    meta = _xmeta(X_all)
    beg, end, dest = _c(beg, torch.int32), _c(end, torch.int32), _c(dest, torch.int32)
    fm = _c(fmask)
    assert fm.shape == (nb, d) and fm.dtype == torch.bool
    narrow = bool(narrow) and BH_RCH <= 65536
    cellbytes = 16 if (wide and not narrow) else 8
    smem_bytes = BH_SMEM_KB * 1024
    smem_cells = smem_bytes // 8
    fgs, ng = meta.groups(smem_bytes // cellbytes)
    if cums is not None:
        assert sizes is not None
        cum_tiny, cum_ch = _c(cums[0], torch.int64), _c(cums[1], torch.int64)
        if DEBUG_CHECKS:
            ln = (end - beg).to(torch.int64)
            thr = max(1, int(math.ceil(BH_SHARED_FACTOR * meta.mean_nbins)))
            is_sh = ln >= thr
            zero = torch.zeros((), dtype=torch.int64, device=dev)
            assert torch.equal(cum_tiny, torch.cumsum(torch.where(is_sh, zero, ln), 0))
            assert torch.equal(cum_ch, torch.cumsum(torch.where(is_sh, (ln + (BH_RCH - 1)) // BH_RCH, zero), 0))
    else:
        ln = (end - beg).to(torch.int64)
        thr = max(1, int(math.ceil(BH_SHARED_FACTOR * meta.mean_nbins)))
        is_sh = ln >= thr
        zero = torch.zeros((), dtype=torch.int64, device=dev)
        cum_tiny = torch.cumsum(torch.where(is_sh, zero, ln), 0)
        cum_ch = torch.cumsum(torch.where(is_sh, (ln + (BH_RCH - 1)) // BH_RCH, zero), 0)
    n_tiny_rows, n_chunks = sizes if sizes is not None else torch.stack([cum_tiny[-1], cum_ch[-1]]).tolist()
    n_tiny_blocks = (n_tiny_rows + BH_RCH - 1) // BH_RCH
    if n_tiny_blocks + n_chunks == 0:
        return
    plane = out.shape[1] * d * B if wide else 0
    wl32, xrow32, gh64 = _c(wl, torch.int32), _c(xrow, torch.int32), _c(gh, torch.int64)
    rc = lib.mg_build_hist(_ptr(X_all), _ptr(xrow32), _ptr(gh64), _ptr(wl32), _ptr(beg), _ptr(end), _ptr(dest), _ptr(fm),
                           _ptr(out), plane, _ptr(cum_tiny), _ptr(cum_ch), _ptr(fgs), _ptr(meta.fbo),
                           nb, d, B, ng, BH_RCH, 1 if wide else 0, n_tiny_rows, n_tiny_blocks, n_chunks,
                           _stream(dev), _threads(dev, BH_THREADS), smem_bytes, 1 if narrow else 0)
    _check(rc, "mg_build_hist")


def node_sums(gh, wl, beg, end, cum=None, n_chunks=None):
    nb = len(beg)
    dev = gh.device
    G = torch.zeros(nb, dtype=torch.int64, device=dev)
    H = torch.zeros(nb, dtype=torch.int64, device=dev)
    if nb == 0:
        return G, H
    lib = _lib(dev)
    beg, end = _c(beg, torch.int32), _c(end, torch.int32)
    if cum is not None:
        cum = _c(cum, torch.int64)
        if DEBUG_CHECKS:
            c2, n2 = _chunks(beg, end, SEG_RCH)
            assert torch.equal(cum, c2) and n2 == n_chunks
    else:
        cum, n_chunks = _chunks(beg, end, SEG_RCH)
    if n_chunks == 0:
        return G, H
    rc = lib.mg_node_sums(_ptr(_c(gh, torch.int64)), _ptr(_c(wl, torch.int32)), _ptr(beg), _ptr(end), _ptr(cum), nb, SEG_RCH,
                          n_chunks, _ptr(G), _ptr(H), _stream(dev), _threads(dev, THREADS))
    _check(rc, "mg_node_sums")
    return G, H


if HAVE_TRITON:

    @triton.jit
    def _sub_kernel(out_ptr, oidx_ptr, par_ptr, pidx_ptr, sm_ptr, sidx_ptr, n_cells, cs_out, cs_par, cs_sm,
                    plane_out, plane_par, plane_sm, PLANES: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        offs = cb * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_cells
        oi = tl.load(oidx_ptr + i).to(tl.int64) * cs_out
        pi = tl.load(pidx_ptr + i).to(tl.int64) * cs_par
        si = tl.load(sidx_ptr + i).to(tl.int64) * cs_sm
        for p in tl.static_range(PLANES):
            a = tl.load(par_ptr + p * plane_par + pi + offs, mask=m, other=0)
            b = tl.load(sm_ptr + p * plane_sm + si + offs, mask=m, other=0)
            tl.store(out_ptr + p * plane_out + oi + offs, a - b, mask=m)

    @triton.jit
    def _decode_kernel(h_ptr, g_out, h_out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(h_ptr + offs, mask=m, other=0)
        h = v & 0xFFFFFFFF
        g = (v - h) >> 32
        tl.store(g_out + offs, g, mask=m)
        tl.store(h_out + offs, h, mask=m)

    @triton.jit
    def _split_kernel(G_ptr, H_ptr, Gt_ptr, Ht_ptr, isg_ptr, ish_ptr, lam_ptr, mcw_ptr, fm_ptr,
                      og_ptr, of_ptr, ob_ptr, ogl_ptr, ohl_ptr, d, NFB,
                      B: tl.constexpr, BLOCK_F: tl.constexpr):
        c = tl.program_id(0).to(tl.int64)
        fb = tl.program_id(1)
        feats = fb * BLOCK_F + tl.arange(0, BLOCK_F)
        fvalid = feats < d
        bins = tl.arange(0, B)
        off = (c * d + feats[:, None]) * B + bins[None, :]
        G = tl.load(G_ptr + off, mask=fvalid[:, None], other=0)
        H = tl.load(H_ptr + off, mask=fvalid[:, None], other=0)
        Gt = tl.load(Gt_ptr + c)
        Ht = tl.load(Ht_ptr + c)
        isg = tl.load(isg_ptr + c)
        ish = tl.load(ish_ptr + c)
        lam = tl.load(lam_ptr + c)
        mcw = tl.load(mcw_ptr + c)
        fm = tl.load(fm_ptr + c * d + feats, mask=fvalid, other=0) != 0
        GL = tl.cumsum(G, axis=1)
        HL = tl.cumsum(H, axis=1)
        GR = Gt - GL
        HR = Ht - HL
        gl = GL.to(tl.float32) * isg
        hl = HL.to(tl.float32) * ish
        gr = GR.to(tl.float32) * isg
        hr = HR.to(tl.float32) * ish
        gt = Gt.to(tl.float32) * isg
        ht = Ht.to(tl.float32) * ish
        parent = tl.div_rn(gt * gt, ht + lam)
        tl_ = tl.div_rn(gl * gl, hl + lam)
        tr_ = tl.div_rn(gr * gr, hr + lam)
        gain = tl_ + tr_ - parent
        ok = (hl >= mcw) & (hr >= mcw) & (HL > 0) & (HR > 0) & fm[:, None]
        gain = tl.where(ok, gain, float("-inf"))
        bbest = tl.max(gain, axis=1)
        bbin = tl.argmax(gain, axis=1, tie_break_left=True)
        fbest = tl.max(bbest, axis=0)
        fidx = tl.argmax(bbest, axis=0, tie_break_left=True)
        selb = bins[None, :] == bbin[:, None]
        glb = tl.sum(tl.where(selb, GL, 0), axis=1)
        hlb = tl.sum(tl.where(selb, HL, 0), axis=1)
        self_f = tl.arange(0, BLOCK_F) == fidx
        GLb = tl.sum(tl.where(self_f, glb, 0), axis=0)
        HLb = tl.sum(tl.where(self_f, hlb, 0), axis=0)
        binb = tl.sum(tl.where(self_f, bbin, 0), axis=0)
        o = c * NFB + fb
        tl.store(og_ptr + o, fbest)
        tl.store(of_ptr + o, (fb * BLOCK_F + fidx).to(tl.int32))
        tl.store(ob_ptr + o, binb.to(tl.int32))
        tl.store(ogl_ptr + o, GLb)
        tl.store(ohl_ptr + o, HLb)

    @triton.jit
    def _rint(x):
        return (x + 12582912.0) - 12582912.0

    @triton.jit
    def _quant_kernel(g_ptr, h_ptr, slot_ptr, work_ptr, sg32_ptr, sg28_ptr, sh32_ptr, sh28_ptr, is28_ptr, gh_ptr, n,
                      HAS_HESS: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        w = tl.load(work_ptr + offs, mask=m, other=0) != 0
        slot = tl.load(slot_ptr + offs, mask=m, other=0).to(tl.int64)
        g = tl.load(g_ptr + offs, mask=w, other=0.0)
        is28 = tl.load(is28_ptr + slot, mask=m, other=0) != 0
        sg32 = tl.load(sg32_ptr + slot, mask=m, other=1.0)
        sg28 = tl.load(sg28_ptr + slot, mask=m, other=1.0)
        r16 = _rint(g * sg32)
        gq16 = tl.minimum(tl.maximum(r16, -32767.0), 32767.0).to(tl.int64)
        gs = g.to(tl.float64) * sg28
        t = gs + tl.where(gs >= 0, 0.5, -0.5)
        gq28 = tl.minimum(tl.maximum(t.to(tl.int64), -134217727), 134217727)
        gq = tl.where(is28, gq28, gq16)
        if HAS_HESS:
            h = tl.load(h_ptr + offs, mask=w, other=0.0)
            sh32 = tl.load(sh32_ptr + slot, mask=m, other=1.0)
            sh28 = tl.load(sh28_ptr + slot, mask=m, other=1.0)
            rh = _rint(h * sh32)
            hq16 = tl.minimum(tl.maximum(rh, 0.0), 65535.0).to(tl.int64)
            hs = h.to(tl.float64) * sh28 + 0.5
            hq28 = tl.minimum(tl.maximum(hs.to(tl.int64), 0), 268435455)
            hq = tl.where(is28, hq28, hq16)
        else:
            hq = tl.full([BLOCK], 1, tl.int64)
        gh = tl.where(w, gq * 4294967296 + hq, 0)
        tl.store(gh_ptr + offs, gh, mask=m)


def _slab_strides(t, wide, d, B, name):
    assert t.shape[-2:] == (d, B), f"{name}: trailing dims {tuple(t.shape[-2:])} != ({d}, {B})"
    assert t.stride(-1) == 1 and t.stride(-2) == B, f"{name}: (d,B) slabs must be contiguous, strides {t.stride()}"
    if wide:
        assert t.dim() == 4 and t.shape[0] == 2, f"{name}: wide histogram must be [2,C,d,B]"
        return t.stride(0), t.stride(1)
    assert t.dim() == 3, f"{name}: packed histogram must be [C,d,B]"
    return 0, t.stride(0)


def subtract_into(out, out_idx, parent, parent_idx, small, small_idx, wide):
    n = len(out_idx)
    if n == 0:
        return
    _require_triton()
    d, B = out.shape[-2], out.shape[-1]
    plane_out, cs_out = _slab_strides(out, wide, d, B, "out")
    plane_par, cs_par = _slab_strides(parent, wide, d, B, "parent")
    plane_sm, cs_sm = _slab_strides(small, wide, d, B, "small")
    n_cells = d * B
    BLOCK = 1024
    oi, pi, si = _c(out_idx, torch.int64), _c(parent_idx, torch.int64), _c(small_idx, torch.int64)
    _sub_kernel[(n, triton.cdiv(n_cells, BLOCK))](out, oi, parent, pi, small, si, n_cells, cs_out, cs_par, cs_sm,
                                                  plane_out, plane_par, plane_sm,
                                                  PLANES=2 if wide else 1, BLOCK=BLOCK, enable_fp_fusion=False)


def decode_hist(hist, wide):
    if wide:
        return hist[0], hist[1]
    _require_triton()
    h = hist.contiguous()
    G = torch.empty_like(h)
    H = torch.empty_like(h)
    n = h.numel()
    if n:
        BLOCK = 1024
        _decode_kernel[(triton.cdiv(n, BLOCK),)](h, G, H, n, BLOCK=BLOCK, enable_fp_fusion=False)
    return G, H


def split_search(G, H, Gt, Ht, isg, ish, lam, mcw, fmask):
    C, d, B = G.shape
    dev = G.device
    if C == 0:
        z = torch.zeros(0, dtype=torch.int32, device=dev)
        return z, z.clone(), torch.zeros(0, dtype=torch.float32, device=dev), torch.zeros(0, dtype=torch.int64, device=dev), torch.zeros(0, dtype=torch.int64, device=dev)
    _require_triton()
    assert B & (B - 1) == 0 and B >= 2
    G, H = G.contiguous(), H.contiguous()
    Gt, Ht = _c(Gt, torch.int64), _c(Ht, torch.int64)
    isg, ish, lam, mcw = _c(isg, torch.float32), _c(ish, torch.float32), _c(lam, torch.float32), _c(mcw, torch.float32)
    fm = _c(fmask).view(torch.uint8)
    assert fm.shape == (C, d)
    BLOCK_F = 8 if B >= 128 else 16
    NFB = triton.cdiv(d, BLOCK_F)
    og = torch.empty((C, NFB), dtype=torch.float32, device=dev)
    of = torch.empty((C, NFB), dtype=torch.int32, device=dev)
    ob = torch.empty((C, NFB), dtype=torch.int32, device=dev)
    ogl = torch.empty((C, NFB), dtype=torch.int64, device=dev)
    ohl = torch.empty((C, NFB), dtype=torch.int64, device=dev)
    _split_kernel[(C, NFB)](G, H, Gt, Ht, isg, ish, lam, mcw, fm, og, of, ob, ogl, ohl, d, NFB,
                            B=B, BLOCK_F=BLOCK_F, enable_fp_fusion=False)
    if NFB == 1:
        return of[:, 0].contiguous(), ob[:, 0].contiguous(), og[:, 0].contiguous(), ogl[:, 0].contiguous(), ohl[:, 0].contiguous()
    best = og.max(dim=1).values
    j = (og == best[:, None]).to(torch.int8).argmax(dim=1)
    j = j[:, None]
    return (of.gather(1, j).squeeze(1), ob.gather(1, j).squeeze(1), og.gather(1, j).squeeze(1),
            ogl.gather(1, j).squeeze(1), ohl.gather(1, j).squeeze(1))


def _slot_maxima(g, h, slot_of_rt, work, S, has_hess, rt_off):
    dev = g.device
    RT = g.numel()
    mode = QMAX
    if mode == "triton" and RT and S:
        gmax = torch.zeros(S, dtype=torch.float32, device=dev)
        hmax = torch.zeros(S, dtype=torch.float32, device=dev) if has_hess else gmax
        BLOCK = 1024
        _amax_kernel[(triton.cdiv(RT, BLOCK),)](g.contiguous(), h.contiguous(), _c(slot_of_rt, torch.int32),
                                                 _c(work).view(torch.uint8), gmax, hmax, RT,
                                                 HAS_HESS=bool(has_hess), BLOCK=BLOCK)
        return gmax, (hmax if has_hess else None)
    off = rt_off if mode == "segment" else None
    zero = torch.zeros((), dtype=torch.float32, device=dev)
    gmax = KR.slot_amax(torch.where(work, g.abs(), zero), slot_of_rt, S, off)
    hmax = KR.slot_amax(torch.where(work, h, zero), slot_of_rt, S, off) if has_hess else None
    return gmax, hmax


QMAX = os.environ.get("MANGROVE_QMAX", "segment")
if QMAX not in ("scatter", "segment", "triton"):
    raise ValueError(f"MANGROVE_QMAX={QMAX!r}: expected scatter | segment | triton")

if HAVE_TRITON:
    @triton.jit
    def _amax_kernel(g_ptr, h_ptr, slot_ptr, work_ptr, gmax_ptr, hmax_ptr, n,
                     HAS_HESS: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        w = m & (tl.load(work_ptr + offs, mask=m, other=0) != 0)
        slot = tl.load(slot_ptr + offs, mask=m, other=0)
        ga = tl.where(w, tl.abs(tl.load(g_ptr + offs, mask=w, other=0.0)), 0.0)
        if HAS_HESS:
            ha = tl.where(w, tl.load(h_ptr + offs, mask=w, other=0.0), 0.0)
        lo = tl.min(tl.where(m, slot, 2147483647), axis=0)
        hi = tl.max(tl.where(m, slot, -1), axis=0)
        for s in range(lo, hi + 1):
            sel = slot == s
            tl.atomic_max(gmax_ptr + s, tl.max(tl.where(sel, ga, 0.0), axis=0))
            if HAS_HESS:
                tl.atomic_max(hmax_ptr + s, tl.max(tl.where(sel, ha, 0.0), axis=0))


def quantize(g, h, slot_of_rt, work, bits, S, has_hess=True, rt_off=None):
    _require_triton()
    dev = g.device
    RT = g.numel()
    gmax, hmax = _slot_maxima(g, h, slot_of_rt, work, S, has_hess, rt_off)
    one32 = torch.ones_like(gmax)
    sg32 = torch.where(gmax > 0, _dconst(G16, torch.float32, dev) / gmax, one32)
    g64 = gmax.to(torch.float64)
    sg28 = torch.where(g64 > 0, G28 / g64, torch.ones_like(g64))
    is28 = (bits == 28)
    sg = torch.where(is28, sg28, sg32.to(torch.float64))
    if has_hess:
        sh32 = torch.where(hmax > 0, _dconst(H16, torch.float32, dev) / hmax, one32)
        h64 = hmax.to(torch.float64)
        sh28 = torch.where(h64 > 0, H28 / h64, torch.ones_like(h64))
        sh = torch.where(is28, sh28, sh32.to(torch.float64))
    else:
        sh = torch.ones(S, dtype=torch.float64, device=dev)
        sh32, sh28 = one32, torch.ones_like(g64)
    gh = torch.empty(RT, dtype=torch.int64, device=dev)
    if RT:
        BLOCK = 1024
        _quant_kernel[(triton.cdiv(RT, BLOCK),)](g.contiguous(), h.contiguous(), _c(slot_of_rt, torch.int32),
                                                  _c(work).view(torch.uint8), sg32.contiguous(), sg28.contiguous(),
                                                  sh32.contiguous(), sh28.contiguous(), _c(is28).view(torch.uint8), gh, RT,
                                                  HAS_HESS=bool(has_hess), BLOCK=BLOCK, enable_fp_fusion=False)
    return gh, sg, sh


def partition(X_all, xrow, wl, beg, end, feat, thr, sizes=None, cums=None):
    P = len(beg)
    if P == 0:
        return beg.clone()
    dev = X_all.device
    lib = _lib(dev)
    assert wl.is_contiguous() and wl.dtype == torch.int32, "partition updates wl in place: int32 contiguous"
    beg, end = _c(beg, torch.int32), _c(end, torch.int32)
    feat, thr = _c(feat, torch.int32), _c(thr, torch.int32)
    if cums is not None:
        assert sizes is not None
        cum_ch, cum_len = _c(cums[0], torch.int64), _c(cums[1], torch.int64)
        if DEBUG_CHECKS:
            ln = (end - beg).to(torch.int64)
            assert torch.equal(cum_ch, torch.cumsum((ln + (SEG_RCH - 1)) // SEG_RCH, 0))
            assert torch.equal(cum_len, torch.cumsum(ln, 0))
    else:
        ln = (end - beg).to(torch.int64)
        cum_ch = torch.cumsum((ln + (SEG_RCH - 1)) // SEG_RCH, 0)
        cum_len = torch.cumsum(ln, 0)
    n_chunks, T = sizes if sizes is not None else torch.stack([cum_ch[-1], cum_len[-1]]).tolist()
    mid = beg.clone()
    if n_chunks == 0:
        return mid
    cnt = torch.empty(n_chunks, dtype=torch.int32, device=dev)
    tmp = torch.empty(max(T, 1), dtype=torch.int32, device=dev)
    fl = torch.empty(max(T, 1), dtype=torch.uint8, device=dev)
    rc = lib.mg_partition(_ptr(X_all), _ptr(_c(xrow, torch.int32)), _ptr(wl), _ptr(beg), _ptr(end), _ptr(feat), _ptr(thr),
                          _ptr(cum_ch), _ptr(cum_len), P, X_all.shape[1], SEG_RCH, n_chunks, _ptr(cnt), _ptr(tmp), _ptr(mid),
                          _stream(dev), _threads(dev, THREADS), _ptr(fl))
    _check(rc, "mg_partition")
    return mid


def leaf_update(pred, wl, beg, end, lv):
    nb = len(beg)
    if nb == 0:
        return
    dev = pred.device
    lib = _lib(dev)
    assert pred.is_contiguous() and pred.dtype == torch.float32
    beg, end = _c(beg, torch.int32), _c(end, torch.int32)
    cum, n_chunks = _chunks(beg, end, SEG_RCH)
    if n_chunks == 0:
        return
    rc = lib.mg_leaf_update(_ptr(pred), _ptr(_c(wl, torch.int32)), _ptr(beg), _ptr(end), _ptr(cum), _ptr(_c(lv, torch.float32)),
                            nb, SEG_RCH, n_chunks, _stream(dev), _threads(dev, THREADS))
    _check(rc, "mg_leaf_update")


ROUTE_ERR_DEFER = True


def route_update(X_all, xrow, pred, q, slot_of_q, levels, leaf_key, leaf_val, node_shift, err_out=None):
    R = len(q)
    if R == 0:
        return
    dev = pred.device
    lib = _lib(dev)
    assert pred.is_contiguous() and pred.dtype == torch.float32
    L = len(levels)
    if L:
        keys = torch.cat([k for k, _, _ in levels]).to(torch.int64).contiguous()
        feat = torch.cat([f for _, f, _ in levels]).to(torch.int32).contiguous()
        thr = torch.cat([t for _, _, t in levels]).to(torch.int32).contiguous()
        lev_off = _h2d_async(np.cumsum([0] + [len(k) for k, _, _ in levels]).astype(np.int32), dev)
    else:
        keys = torch.zeros(0, dtype=torch.int64, device=dev)
        feat = torch.zeros(0, dtype=torch.int32, device=dev)
        thr = feat.clone()
        lev_off = torch.zeros(1, dtype=torch.int32, device=dev)
    lk, lv = _c(leaf_key, torch.int64), _c(leaf_val, torch.float32)
    err = torch.zeros(1, dtype=torch.int32, device=dev)
    rc = lib.mg_route_update(_ptr(X_all), _ptr(_c(xrow, torch.int32)), _ptr(pred), _ptr(_c(q, torch.int64)),
                             _ptr(_c(slot_of_q, torch.int64)), R, _ptr(lev_off), _ptr(keys), _ptr(feat), _ptr(thr), L,
                             _ptr(lk), _ptr(lv), len(lk), int(node_shift), X_all.shape[1], _ptr(err),
                             _stream(dev), _threads(dev, THREADS))
    _check(rc, "mg_route_update")
    if err_out is not None:
        err_out.append(err)
        return
    n_err = int(err.item())
    assert n_err == 0, f"route_update: {n_err} routed rows did not land on a leaf"


__all__ = ["mix32", "hash_u01", "grad_hess", "quantize", "split_gh", "build_hist", "build_hist_sizes", "partition_sizes",
           "build_hist_grid", "partition_grid", "segment_grid", "node_sums", "subtract_into",
           "decode_hist", "split_search", "partition", "leaf_values", "leaf_update", "route_update", "metric_fixed_point",
           "G16", "H16", "G28", "H28", "KRTEPS", "NEG_INF", "_expand_segments", "EMU", "HAVE_TRITON"]
