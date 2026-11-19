import ctypes
import math
import os

import numpy as np
import torch

from . import kernels_gpu as KG

NM_THREADS = int(os.environ.get("MGNM_THREADS", "512"))
HIST_THREADS = min(NM_THREADS, 512)
HIST_AUTO = "MGNM_THREADS" not in os.environ
TABLE_KB_ROOT = int(os.environ.get("MGNM_TABLE_KB_ROOT", "48"))
TABLE_KB = int(os.environ.get("MGNM_TABLE_KB", "32"))
BLOCKS_PER_SM = int(os.environ.get("MGNM_BLOCKS_PER_SM", "4"))
FORCE_RCH = int(os.environ.get("MGNM_RCH", "0"))
EXPECTED_NM_VERSION = 9
NM_MAXB = 128
NM_MAXK = 32
_TRUE1 = np.ones(1, dtype=bool)
FINISH_THREADS = 256
SCALES_THREADS = 256
GRAD_THREADS = 256
ROWMASK_THREADS = 256
ROW_STREAM, COL_STREAM = 1, 2
SCAL_PAD = 32
FG_CHOICES = (32, 28, 16, 8, 4, 2, 1)
STATIC_SMEM = 2048

_READY = set()
_SM = {}


def lib(dev):
    L = KG._lib(dev)
    if id(L) in _READY:
        return L
    P, I, Lg = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    L.mgnm_version.restype = I
    L.mgnm_version.argtypes = []
    v = L.mgnm_version()
    if v != EXPECTED_NM_VERSION:
        raise RuntimeError(f"mgnm_version() = {v}, kernels_nm.py expects {EXPECTED_NM_VERSION}: rebuild cuda/")
    L.mgnm_hist.restype = I
    L.mgnm_hist.argtypes = [P] * 5 + [Lg, I, P, I, I, P, P] + [I] * 8 + [Lg, P, I, I, I]
    L.mgnm_update.restype = I
    L.mgnm_update.argtypes = [P] * 4 + [I, I] + [P] * 3 + [I] * 3 + [Lg, P, I]
    L.mgnm_leaf.restype = I
    L.mgnm_leaf.argtypes = [P] * 3 + [I, I] + [P] * 2 + [I] + [Lg, P, I]
    L.mgnm_compact.restype = I
    L.mgnm_compact.argtypes = [P] * 2 + [I, I] + [P] * 4 + [I, I] + [Lg, P, I]
    L.mgnm_split_finish.restype = I
    L.mgnm_split_finish.argtypes = [P] * 5 + [I, I, P, Lg, Lg, P, I]
    L.mgnm_scales.restype = I
    L.mgnm_scales.argtypes = [P] * 3 + [I, I] + [P] * 7 + [P, I]
    L.mgnm_rowsum.restype = I
    L.mgnm_rowsum.argtypes = [P] * 4 + [I] + [P] * 3 + [I, Lg, P, I]
    L.mgnm_zero.restype = I
    L.mgnm_zero.argtypes = [P, Lg, P]
    L.mgnm_grad.restype = I
    L.mgnm_grad.argtypes = [I, I] + [P] * 6 + [Lg, Lg, P, I]
    L.mgnm_rowmask.restype = I
    L.mgnm_rowmask.argtypes = [P] * 4 + [I] + [P] * 2 + [Lg, P, I]
    _READY.add(id(L))
    return L


def available(dev):
    if dev.type != "cuda":
        return False
    try:
        lib(dev)
        return True
    except Exception:
        return False


def _p(t):
    return t.data_ptr()


def _dev_index(dev):
    k = dev.index
    return torch._C._cuda_getDevice() if k is None else k


def _stream(dev):
    return torch._C._cuda_getCurrentRawStream(_dev_index(dev))


def _check(rc, name):
    if rc != 0:
        raise RuntimeError(f"{name} failed with code {rc} (negative = argument check in mg_nodemap.cu, positive = cudaError_t)")


def _c(t, dtype):
    return t if (t.dtype == dtype and t.is_contiguous()) else t.to(dtype).contiguous()


def sm_info(dev):
    k = _dev_index(dev)
    r = _SM.get(k)
    if r is None:
        p = torch.cuda.get_device_properties(k)
        r = _SM[k] = (p.multi_processor_count, int(getattr(p, "shared_memory_per_block_optin", 101376)))
    return r


class Meta:
    def __init__(self, X_all):
        m = KG._xmeta(X_all)
        self.nbins = m.nbins
        self.fbo = m.fbo
        self.d = len(self.nbins)
        self.words = self.d % 4 == 0 and X_all.is_contiguous()
        self._w = {}

    def wmax(self, FG):
        if FG not in self._w:
            self._w[FG] = max(sum(self.nbins[f0:f0 + FG]) for f0 in range(0, self.d, FG))
        return self._w[FG]


_SHAPE_MEMO = {}


def _memo_put(key, val):
    if len(_SHAPE_MEMO) > 8192:
        _SHAPE_MEMO.clear()
    _SHAPE_MEMO[key] = val
    return val


def light_rch(dev, rows):
    key = ("rch", _dev_index(dev), rows)
    r = _SHAPE_MEMO.get(key)
    if r is not None:
        return r
    sms, _ = sm_info(dev)
    per = max(1, rows // max(1, sms * 8))
    return _memo_put(key, max(1024, min(16384, 1 << int(math.floor(math.log2(per))))))


def hist_shape(dev, meta, MN, rows):
    key = ("shape", _dev_index(dev), meta, MN, rows, TABLE_KB_ROOT, TABLE_KB, STATIC_SMEM, FG_CHOICES, FORCE_RCH,
           BLOCKS_PER_SM)
    r = _SHAPE_MEMO.get(key)
    if r is None:
        r = _memo_put(key, _hist_shape(dev, meta, MN, rows))
    return dict(r)


def _hist_shape(dev, meta, MN, rows):
    _, optin = sm_info(dev)
    budget = min(((TABLE_KB_ROOT if MN == 1 else TABLE_KB) * 1024) // 8, (optin - STATIC_SMEM) // 8)
    FG, MNb, nnb = None, MN, 1
    for f in FG_CHOICES:
        if meta.words and f >= 4 and f % 4:
            continue
        if MN * meta.wmax(f) <= budget:
            FG = f
            break
    if FG is None:
        FG = 1
        MNb = max(1, budget // meta.wmax(1))
        nnb = (MN + MNb - 1) // MNb
    ng = (meta.d + FG - 1) // FG
    if FORCE_RCH:
        rch = FORCE_RCH
    else:
        sms, _ = sm_info(dev)
        per = max(1, rows * ng * nnb // max(1, sms * BLOCKS_PER_SM))
        rch = max(2048, min(65536, 1 << int(math.floor(math.log2(per)))))
    return dict(FG=FG, MNb=MNb, nnb=nnb, smem=MNb * meta.wmax(FG) * 8, rch=rch, ng=ng)


class Plan:
    def __init__(self, rt_off_h, slots, nodes=None, aux=None):
        slots = np.asarray(slots, dtype=np.int64)
        z = np.zeros(0, dtype=np.int64)
        if nodes is None:
            self.cs = slots
            self.order = z
            self.eoff = np.zeros(len(slots) + 1, dtype=np.int64)
            self.node = self.aux = self.aux2 = self.ent_slot = z
            self.MN = 0
        else:
            n = len(slots)
            if n and (n == 1 or bool((slots == slots[0]).all())):
                order = np.arange(n, dtype=np.intp)
                ss = slots.copy()
                self.cs = ss[:1].copy()
                self.eoff = np.array([0, n], dtype=np.int64)
                self.order = order
                self.node = np.asarray(nodes, dtype=np.int64)[order]
                self.aux = np.zeros(n, dtype=np.int64) if aux is None else np.asarray(aux, dtype=np.int64)[order]
                self.aux2 = np.zeros(n, dtype=np.int64)
                self.ent_slot = ss
                self.MN = n
            else:
                order = np.argsort(slots, kind="stable")
                ss = slots[order]
                starts = np.flatnonzero(np.concatenate((_TRUE1, ss[1:] != ss[:-1]))) if len(ss) else z
                self.cs = ss[starts]
                self.eoff = np.append(starts, len(ss)).astype(np.int64)
                self.order = order
                self.node = np.asarray(nodes, dtype=np.int64)[order]
                self.aux = np.zeros(len(ss), dtype=np.int64) if aux is None else np.asarray(aux, dtype=np.int64)[order]
                self.aux2 = np.zeros(len(ss), dtype=np.int64)
                self.ent_slot = ss
                self.MN = int(np.diff(self.eoff).max()) if len(ss) else 0
            if self.MN > NM_MAXB:
                raise RuntimeError(f"node-map call with {self.MN} entries in one slot (max {NM_MAXB})")
            if len(ss) and (int(self.node.min()) < 0 or int(self.node.max()) >= 255):
                raise RuntimeError("node-map heap id outside [0, 255)")
        self.ns = len(self.cs)
        self.E = len(self.node)
        self.a = rt_off_h[self.cs]
        self.b = rt_off_h[self.cs + 1]
        self.rows = int((self.b - self.a).sum())
        self.dv = None

    def upload(self, h2d, rch, xbase_all=None, extra=None):
        nch = (self.b - self.a + rch - 1) // rch
        self.rch = int(rch)
        self.n_chunks = int(nch.sum())
        xb = np.full(self.ns, -1, dtype=np.int64) if xbase_all is None else xbase_all[self.cs]
        words = np.concatenate([self.a, self.b, np.cumsum(nch), xb, self.cs, self.eoff, self.node, self.aux,
                                self.aux2, self.ent_slot])
        if extra is None:
            self.dv = h2d(words)
            return self.dv
        views = h2d([(words, "i64")] + list(extra))
        self.dv = views[0]
        return views[1:]

    def ent_slot_dev(self):
        o = 6 * self.ns + 1 + 3 * self.E
        return self.dv[o:o + self.E]


def hist(X_all, xrow, gh, nm, out, wide, P, shape, fmask_u8, meta):
    dev = X_all.device
    d, B = X_all.shape[1], out.shape[-1]
    plane = out.shape[1] * d * B if wide else 0
    assert fmask_u8.dtype == torch.uint8 and fmask_u8.is_contiguous() and fmask_u8.shape[1] == d
    rc = lib(dev).mgnm_hist(_p(X_all), _p(xrow), _p(gh), _p(nm), _p(out), plane, 1 if wide else 0,
                            _p(P.dv), P.ns, P.E, _p(fmask_u8), _p(meta.fbo),
                            d, B, shape["FG"], P.rch, shape["MNb"], shape["nnb"], P.MN, 1 if meta.words else 0,
                            P.n_chunks, _stream(dev), HIST_THREADS, shape["smem"], 1 if HIST_AUTO else 0)
    _check(rc, "mgnm_hist")


def update(X_all, xrow, nm, P, sfeat, sthr):
    dev = X_all.device
    sf, st = _c(sfeat, torch.int32), _c(sthr, torch.int32)
    lcount = torch.empty(max(P.E, 1), dtype=torch.int32, device=dev)
    rc = lib(dev).mgnm_update(_p(X_all), _p(xrow), _p(nm), _p(P.dv), P.ns, P.E, _p(sf), _p(st), _p(lcount),
                              X_all.shape[1], P.rch, max(P.MN, 1), P.n_chunks, _stream(dev), NM_THREADS)
    _check(rc, "mgnm_update")
    return lcount[:P.E]


def leaf(pred, nm, P, lv, err):
    dev = pred.device
    v = _c(lv, torch.float32)
    rc = lib(dev).mgnm_leaf(_p(pred), _p(nm), _p(P.dv), P.ns, P.E, _p(v), _p(err), P.rch, P.n_chunks, _stream(dev),
                            NM_THREADS)
    _check(rc, "mgnm_leaf")


def compact(nm, P, W, err):
    dev = nm.device
    n = max(P.n_chunks * P.MN, 1)
    ccnt = torch.empty(n, dtype=torch.int32, device=dev)
    cbase = torch.empty(n, dtype=torch.int64, device=dev)
    wl = torch.empty(max(W, 1), dtype=torch.int32, device=dev)
    rc = lib(dev).mgnm_compact(_p(nm), _p(P.dv), P.ns, P.E, _p(ccnt), _p(cbase), _p(wl), _p(err), P.rch, max(P.MN, 1),
                               P.n_chunks, _stream(dev), NM_THREADS)
    _check(rc, "mgnm_compact")
    return wl[:W]


def split_finish(og, of, ob, ogl, ohl, RB, C_chunk, pos0):
    C, NFB = og.shape
    if C == 0:
        return
    dev = og.device
    assert og.dtype == torch.float32 and of.dtype == torch.int32 and ob.dtype == torch.int32
    assert ogl.dtype == torch.int64 and ohl.dtype == torch.int64
    assert all(t.shape == (C, NFB) and t.is_contiguous() for t in (og, of, ob, ogl, ohl))
    assert RB.dtype == torch.int32 and RB.is_contiguous() and RB.device == dev and RB.data_ptr() % 8 == 0
    assert 0 <= pos0 and pos0 + C <= C_chunk and RB.numel() >= 7 * C_chunk, (pos0, C, C_chunk, RB.numel())
    rc = lib(dev).mgnm_split_finish(_p(og), _p(of), _p(ob), _p(ogl), _p(ohl), int(C), int(NFB), _p(RB), int(C_chunk),
                                    int(pos0), _stream(dev), FINISH_THREADS)
    _check(rc, "mgnm_split_finish")


def split_search_nm(G, H, Gt, Ht, isg, ish, lam, mcw, fmask, RB, C_chunk, pos0):
    C, d, B = G.shape
    dev = G.device
    if C == 0:
        return
    KG._require_triton()
    assert B & (B - 1) == 0 and B >= 2
    G, H = G.contiguous(), H.contiguous()
    Gt, Ht = KG._c(Gt, torch.int64), KG._c(Ht, torch.int64)
    isg, ish, lam, mcw = KG._c(isg, torch.float32), KG._c(ish, torch.float32), KG._c(lam, torch.float32), KG._c(mcw, torch.float32)
    fm = KG._c(fmask).view(torch.uint8)
    assert fm.shape == (C, d)
    BLOCK_F = 8 if B >= 128 else 16
    NFB = KG.triton.cdiv(d, BLOCK_F)
    og = torch.empty((C, NFB), dtype=torch.float32, device=dev)
    of = torch.empty((C, NFB), dtype=torch.int32, device=dev)
    ob = torch.empty((C, NFB), dtype=torch.int32, device=dev)
    ogl = torch.empty((C, NFB), dtype=torch.int64, device=dev)
    ohl = torch.empty((C, NFB), dtype=torch.int64, device=dev)
    KG._split_kernel[(C, NFB)](G, H, Gt, Ht, isg, ish, lam, mcw, fm, og, of, ob, ogl, ohl, d, NFB,
                               B=B, BLOCK_F=BLOCK_F, enable_fp_fusion=False)
    split_finish(og, of, ob, ogl, ohl, RB, C_chunk, pos0)


class Scal:
    SECTIONS = (("sg32", torch.float32), ("sh32", torch.float32), ("sg28", torch.float64), ("sh28", torch.float64),
                ("is28", torch.uint8), ("G", torch.int64), ("H", torch.int64), ("cnt", torch.int32),
                ("sg", torch.float64), ("sh", torch.float64))

    def __init__(self, S, dev):
        S = int(S)
        Sp = max(SCAL_PAD, -(-S // SCAL_PAD) * SCAL_PAD)
        self.S, self.Sp, self.dev = S, Sp, dev
        self.w = {}
        off = 0
        for name, dt in self.SECTIONS:
            n = Sp * torch.tensor([], dtype=dt).element_size() // 8
            self.w[name] = (off, off + n)
            off += n
        self.nw = off
        self.buf = torch.zeros(off, dtype=torch.int64, device=dev)
        self.sec = {name: self.buf[a:b].view(dt)[:S] for (name, dt), (a, b) in zip(self.SECTIONS, self.w.values())}
        for name, t in self.sec.items():
            setattr(self, name, t)
        wz0, wz1 = self.w["G"][0], self.w["cnt"][1]
        self.zero_bytes = (wz0 * 8, (wz1 - wz0) * 8)
        self.rb_words = (self.w["G"][0], self.w["sh"][1])

    def views(self, s0, s1):
        assert 0 <= s0 <= s1 <= self.S, (s0, s1, self.S)
        return _Views(**{name: t[s0:s1] for name, t in self.sec.items()})

    def zero_sums(self):
        o, n = self.zero_bytes
        if self.dev.type == "cuda":
            _check(lib(self.dev).mgnm_zero(ctypes.c_void_p(self.buf.data_ptr() + o), n, _stream(self.dev)), "mgnm_zero")
        else:
            self.buf[o // 8:(o + n) // 8].zero_()

    def readback(self):
        a, b = self.rb_words
        pk = self.buf[a:b].cpu().numpy()
        if self.dev.type != "cuda":
            pk = pk.copy()
        out = []
        for name, dt in self.SECTIONS[5:]:
            o0, o1 = self.w[name]
            v = pk[o0 - a:o1 - a]
            out.append((v if dt == torch.int64 else v.view({torch.int32: np.int32, torch.float64: np.float64}[dt]))[:self.S])
        return tuple(out)


class _Views:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def scales(gmax, hmax, bits, v, has_hess):
    dev = gmax.device
    S = gmax.numel()
    assert gmax.dtype == torch.float32 and gmax.is_contiguous()
    assert bits.dtype == torch.int8 and bits.is_contiguous() and bits.numel() == S
    if has_hess:
        assert hmax is not None and hmax.dtype == torch.float32 and hmax.is_contiguous() and hmax.numel() == S
        hp = _p(hmax)
    else:
        hp = ctypes.c_void_p(None)
    assert all(t.numel() == S and t.is_contiguous() for t in (v.sg32, v.sh32, v.sg28, v.sh28, v.is28, v.sg, v.sh))
    rc = lib(dev).mgnm_scales(_p(gmax), hp, _p(bits), int(S), 1 if has_hess else 0, _p(v.sg32), _p(v.sh32), _p(v.sg28),
                              _p(v.sh28), _p(v.is28), _p(v.sg), _p(v.sh), _stream(dev), SCALES_THREADS)
    _check(rc, "mgnm_scales")


def quantize_nm(KR, g, h, slot_of_rt, work, bits, S, has_hess, rt_off, scal, s0):
    assert KR is KG, "quantize_nm reproduces kernels_gpu.quantize: the engine backend must be the GPU module"
    KG._require_triton()
    dev = g.device
    RT = g.numel()
    S = int(S)
    gmax, hmax = KG._slot_maxima(g, h, slot_of_rt, work, S, has_hess, rt_off)
    v = scal.views(s0, s0 + S)
    scales(gmax, hmax, bits, v, has_hess)
    gh = torch.empty(RT, dtype=torch.int64, device=dev)
    if RT:
        BLOCK = 1024
        KG._quant_kernel[(KG.triton.cdiv(RT, BLOCK),)](g.contiguous(), h.contiguous(), KG._c(slot_of_rt, torch.int32),
                                                      KG._c(work).view(torch.uint8), v.sg32, v.sg28, v.sh32, v.sh28,
                                                      v.is28, gh, RT, HAS_HESS=bool(has_hess), BLOCK=BLOCK,
                                                      enable_fp_fusion=False)
    return gh, v.sg, v.sh


def rowsum(gh, work, nm, P, scal):
    dev = gh.device
    assert P.ns == scal.S and P.E == 0 and P.dv is not None, "rowsum: a slots-only descriptor of every slot"
    assert gh.dtype == torch.int64 and gh.is_contiguous() and work.dtype == torch.bool and work.is_contiguous()
    assert nm.dtype == torch.uint8 and nm.is_contiguous() and nm.numel() == work.numel() == gh.numel()
    assert P.rows == nm.numel() and (P.ns == 0 or int(P.a[0]) == 0), "rowsum: the descriptor must cover every routed row"
    rc = lib(dev).mgnm_rowsum(_p(gh), _p(work), _p(nm), _p(P.dv), P.ns, _p(scal.G), _p(scal.H), _p(scal.cnt), P.rch,
                              P.n_chunks, _stream(dev), NM_THREADS)
    _check(rc, "mgnm_rowsum")


def route_update_nm(X_all, xrow, pred, q, slot_of_q, lev_off, keys, feat, thr, L, lkey_s, lv_s, node_shift, err):
    R = len(q)
    if R == 0:
        return
    dev = pred.device
    assert pred.is_contiguous() and pred.dtype == torch.float32
    assert keys.dtype == torch.int64 and lkey_s.dtype == torch.int64 and lv_s.dtype == torch.float32
    assert feat.dtype == torch.int32 and thr.dtype == torch.int32 and lev_off.dtype == torch.int32
    assert len(lev_off) == L + 1 and len(feat) == len(keys) == len(thr) and len(lv_s) == len(lkey_s)
    assert all(t.is_contiguous() for t in (keys, feat, thr, lev_off, lkey_s, lv_s))
    L_ = KG._lib(dev)
    rc = L_.mg_route_update(_p(X_all), _p(KG._c(xrow, torch.int32)), _p(pred), _p(KG._c(q, torch.int64)),
                            _p(KG._c(slot_of_q, torch.int64)), R, _p(lev_off), _p(keys), _p(feat), _p(thr), int(L),
                            _p(lkey_s), _p(lv_s), len(lkey_s), int(node_shift), X_all.shape[1], _p(err),
                            _stream(dev), KG.THREADS)
    KG._check(rc, "mg_route_update")


def grad_hess_nm(loss, K, pred, y, mrow_base, mrow_len):
    if loss == "logistic":
        li = 0
    elif loss == "softmax":
        li = 1
    else:
        return None
    if pred.device.type != "cuda" or pred.dtype != torch.float32 or not pred.is_contiguous():
        return None
    n = pred.numel()
    if y.device != pred.device or not y.is_contiguous() or y.numel() != n:
        return None
    if li == 0:
        if y.dtype != torch.float32:
            return None
        Nm, pb, pl = n, 0, 0
    else:
        K = int(K)
        if y.dtype != torch.int64 or K < 1 or K > NM_MAXK or mrow_base is None or mrow_len is None:
            return None
        if mrow_base.dtype != torch.int64 or mrow_len.dtype != torch.int64:
            return None
        if mrow_base.device != pred.device or mrow_len.device != pred.device:
            return None
        if not mrow_base.is_contiguous() or not mrow_len.is_contiguous():
            return None
        Nm = mrow_base.numel()
        if mrow_len.numel() != Nm or Nm * K != n:
            return None
        pb, pl = _p(mrow_base), _p(mrow_len)
    g = torch.empty_like(pred)
    h = torch.empty_like(pred)
    if n == 0:
        return g, h
    dev = pred.device
    rc = lib(dev).mgnm_grad(li, int(K) if li else 1, _p(pred), _p(y), pb, pl, _p(g), _p(h), n, Nm, _stream(dev),
                            GRAD_THREADS)
    _check(rc, "mgnm_grad")
    return g, h


def rowmask_tab(seed_h, lr_h, sub_h, act_h):
    seed_h = np.ascontiguousarray(seed_h, dtype=np.int64)
    S = seed_h.shape[0]
    lr_h = np.ascontiguousarray(lr_h, dtype=np.int64)
    sub_h = np.ascontiguousarray(sub_h, dtype=np.float64)
    act_h = np.asarray(act_h, dtype=bool)
    assert lr_h.shape == sub_h.shape == act_h.shape == (S,), (S, lr_h.shape, sub_h.shape, act_h.shape)
    tab = np.empty(4 * S, dtype=np.int64)
    tab[:S] = seed_h
    tab[S:2 * S] = lr_h
    tab[2 * S:3 * S] = sub_h.view(np.int64)
    tab[3 * S:] = act_h.astype(np.int64) | ((sub_h < 1.0).astype(np.int64) << 1)
    return tab


def rowmask(is_train, crow, slot_of_rt, a, n, tab, S, work, u_out=None):
    dev = work.device
    n, a, S = int(n), int(a), int(S)
    assert is_train.dtype == torch.bool and crow.dtype == torch.int32 and slot_of_rt.dtype == torch.int32
    assert tab.dtype == torch.int64 and tab.numel() == 4 * S and work.dtype == torch.bool and work.numel() == n
    assert all(t.is_contiguous() and t.device == dev for t in (is_train, crow, slot_of_rt, tab, work))
    assert 0 <= a and a + n <= is_train.numel() == crow.numel() == slot_of_rt.numel(), (a, n, is_train.numel())
    if u_out is not None:
        assert u_out.dtype == torch.float64 and u_out.numel() == n and u_out.is_contiguous() and u_out.device == dev
    if n == 0:
        return
    up = ctypes.c_void_p(None) if u_out is None else _p(u_out)
    rc = lib(dev).mgnm_rowmask(_p(is_train) + a, _p(crow) + 4 * a, _p(slot_of_rt) + 4 * a, _p(tab), S, _p(work), up,
                               n, _stream(dev), ROWMASK_THREADS)
    _check(rc, "mgnm_rowmask")


_U32 = np.uint64(0xFFFFFFFF)
_HK1, _HK2, _HK3 = np.uint64(0x85EBCA77), np.uint64(0xC2B2AE3D), np.uint64(0x9E3779B1)
_HM1, _HM2 = np.uint64(0x7FEB352D), np.uint64(0x846CA68B)
_S16, _S15 = np.uint64(16), np.uint64(15)


def _mix32_np(x):
    x = x & _U32
    x = ((x ^ (x >> _S16)) * _HM1) & _U32
    x = ((x ^ (x >> _S15)) * _HM2) & _U32
    return x ^ (x >> _S16)


def hash_u32_np(*keys):
    h = np.zeros((), dtype=np.uint64)
    for k in keys:
        k = np.asarray(k, dtype=np.int64).astype(np.uint64)
        h = _mix32_np(h * _HK3 + _mix32_np(k * _HK1 + _HK2))
    return h


def colsample_host(seed_h, lr_h, level_key, base, frac_h, d=None):
    seed_h = np.asarray(seed_h, dtype=np.int64)
    lr_h = np.asarray(lr_h, dtype=np.int64)
    S = seed_h.shape[0]
    if base is None:
        d = int(d)
        nbase = np.full(S, d, dtype=np.int64)
    else:
        base = np.asarray(base, dtype=bool)
        assert base.shape[0] == S, (base.shape, S)
        d = base.shape[1]
        nbase = base.sum(1)
    h = hash_u32_np(COL_STREAM, seed_h[:, None], lr_h[:, None], int(level_key), np.arange(d, dtype=np.int64)[None, :])
    if base is not None:
        h = np.where(base, h, np.uint64(1) << np.uint64(32))
    with np.errstate(invalid="ignore", over="ignore"):
        frac32 = np.asarray(frac_h, dtype=np.float64).astype(np.float32)
        prod = np.minimum(frac32 * nbase.astype(np.float32), np.float32(d))
        k = np.maximum(np.floor(prod).astype(np.int64), 1)
    order = np.argsort(h, axis=1, kind="stable")
    rank = np.empty((S, d), dtype=np.int64)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(d, dtype=np.int64), (S, d)), axis=1)
    m = rank < k[:, None]
    if base is not None:
        m &= base
    return m.astype(np.uint8)
