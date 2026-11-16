import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch

from util import synth, sample_hpo, make_specs, tree_sig
import mangrove.engine as ENG
from mangrove import Engine, Cohort, ModelSpec
from mangrove import kernels_ref as KR

RESULTS = {"sections": {}, "failures": []}
TABLES = {"binary": (48842, 14, "binary", 1), "multiclass": (20000, 16, "multiclass", 26),
          "regression": (20640, 8, "regression", 1), "binary-wide": (50000, 28, "binary", 1)}
LOSS = {"binary": "logistic", "regression": "squared", "multiclass": "softmax"}
T0 = time.time()
SZ = 1.0
DEV = "cuda"


def table(name, n_max=None, seed=0):
    n, d, task, K = TABLES[name]
    Xb, y, folds = synth(n if n_max is None else min(n, n_max), d, nb=256, seed=seed, task=task, K=K)
    return Xb, y, folds, LOSS[task], K


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def bits_equal(a, b):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(bits_equal(x, y) for x, y in zip(a, b))
    if not torch.is_tensor(a):
        return a == b
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    if a.dtype == torch.float64:
        return torch.equal(a.view(torch.int64), b.view(torch.int64))
    return torch.equal(a, b)


def describe_mismatch(a, b):
    if isinstance(a, (tuple, list)):
        for i, (x, y) in enumerate(zip(a, b)):
            if not bits_equal(x, y):
                return f"output[{i}]: " + describe_mismatch(x, y)
        return "lengths differ"
    if not torch.is_tensor(a):
        return f"{a!r} vs {b!r}"
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
    diff = (a != b) if not a.dtype.is_floating_point else (a.view(torch.int32 if a.dtype == torch.float32 else torch.int64) != b.view(torch.int32 if b.dtype == torch.float32 else torch.int64))
    n = int(diff.sum())
    idx = torch.nonzero(diff)[:5].tolist()
    vals = [(a[tuple(i)].item(), b[tuple(i)].item()) for i in idx]
    return f"{n}/{a.numel()} cells differ; first {idx} ref vs gpu {vals}"


def record(section, name, ok, info="", t=0.0):
    RESULTS["sections"].setdefault(section, []).append(dict(name=name, status="PASS" if ok else "FAIL", info=info, seconds=round(t, 3)))
    if not ok:
        RESULTS["failures"].append(f"{section}/{name}: {info}")
    log(f"{'PASS' if ok else 'FAIL'}  {section:10s} {name}  {info if not ok else ''}")


def run_check(section, name, fn):
    t = time.time()
    try:
        info = fn()
        record(section, name, True, info or "", time.time() - t)
        return True
    except Exception as e:
        record(section, name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", time.time() - t)
        return False


def env_report(dev):
    r = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
             cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), emulation=(DEV != "cuda"))
    try:
        import triton
        r["triton"] = triton.__version__
    except Exception as e:
        r["triton"] = f"MISSING ({e})"
    if DEV == "cuda":
        p = torch.cuda.get_device_properties(dev)
        r["device"] = dict(name=p.name, cc=f"{p.major}.{p.minor}", sms=p.multi_processor_count, mem_gb=round(p.total_memory / 2 ** 30, 2))
    else:
        r["device"] = "cpu emulation (libmangrove_emu.so + TRITON_INTERPRET)"
    try:
        r["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,compute_mode",
                                          "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        r["nvidia_smi_other_processes"] = procs
        if procs:
            log(f"WARNING: other GPU compute processes present:\n{procs}")
    except Exception as e:
        r["nvidia_smi"] = f"unavailable ({e})"
    from mangrove import kernels_gpu as KG
    lib = KG._lib(dev)
    r["kernel_lib"] = dict(path=str(KG.CUDA_DIR), version=lib.mg_version(), tunables=dict(
        BH_RCH=KG.BH_RCH, BH_THREADS=KG.BH_THREADS, BH_SMEM_KB=KG.BH_SMEM_KB, BH_SHARED_FACTOR=KG.BH_SHARED_FACTOR,
        SEG_RCH=KG.SEG_RCH, THREADS=KG.THREADS))
    RESULTS["env"] = r
    log(json.dumps(r, indent=1))
    return r


def rand_gh(rng, n, bits, dev):
    if bits == 16:
        gq = torch.from_numpy(rng.integers(-32767, 32768, n))
        hq = torch.from_numpy(rng.integers(0, 65536, n))
    else:
        gq = torch.from_numpy(rng.integers(-(2 ** 27 - 1), 2 ** 27, n))
        hq = torch.from_numpy(rng.integers(0, 2 ** 28, n))
    return (gq * 4294967296 + hq).to(dev)


def segments(rng, W, nb, dev, nonempty=False):
    if nonempty:
        nb = min(nb, W // 2)
        cuts = np.sort(rng.choice(np.arange(1, W), nb - 1, replace=False))
    else:
        cuts = np.sort(rng.integers(0, W + 1, nb - 1))
    beg = np.concatenate([[0], cuts])
    end = np.concatenate([cuts, [W]])
    return torch.tensor(beg, dtype=torch.int32, device=dev), torch.tensor(end, dtype=torch.int32, device=dev)


def check_narrow_wide_hist(dev, KG):
    old_rch = KG.BH_RCH
    try:
        for rch in dict.fromkeys((old_rch, 65536)):
            KG.BH_RCH = rch
            for n in (65536, 70000):
                X = torch.zeros((n, 2), dtype=torch.uint8, device=dev)
                X[:, 1] = 3
                rows = torch.arange(n, dtype=torch.int32, device=dev)
                beg = torch.tensor([0], dtype=torch.int32, device=dev)
                end = torch.tensor([n], dtype=torch.int32, device=dev)
                dest = torch.tensor([0], dtype=torch.int32, device=dev)
                fm = torch.ones((1, 2), dtype=torch.bool, device=dev)
                for sign in (1, -1):
                    gh = torch.full((n,), sign * 32767 * 4294967296 + 65535,
                                    dtype=torch.int64, device=dev)
                    expected = torch.zeros((2, 1, 2, 4), dtype=torch.int64, device=dev)
                    actual = torch.zeros_like(expected)
                    KR.build_hist(X, rows, gh, rows, beg, end, dest, fm, expected, True)
                    KG.build_hist(X, rows, gh, rows, beg, end, dest, fm, actual, True, narrow=True)
                    assert bits_equal(expected, actual), (
                        f"narrow wide n={n}, rch={rch}, sign={sign}: " + describe_mismatch(expected, actual))
                    assert int(actual[0, 0, 0, 0]) == sign * n * 32767
                    assert int(actual[1, 0, 0, 0]) == n * 65535
        return "wide+narrow: signed extremes, default/65536 chunks, 65536/70000 rows"
    finally:
        KG.BH_RCH = old_rch


def random_kernel_checks(dev, KG, full):
    sec = "random"
    rng = np.random.default_rng(2026)
    scale = SZ

    if full:
        run_check(sec, "build_hist[wide+narrow,boundaries]", lambda: check_narrow_wide_hist(dev, KG))

    def t_build_hist(wide, case):
        d = 28 if case == "higgs_like" else 9
        B = 256
        nbins = [256] * d if case == "higgs_like" else [73, 8, 255, 15, 15, 6, 14, 2, 1]
        N = int(60000 * scale)
        X = torch.from_numpy(np.stack([rng.integers(0, nb, N) for nb in nbins], 1).astype(np.uint8)).to(dev)
        xrow = torch.from_numpy(rng.permutation(N).astype(np.int32)).to(dev)
        gh = rand_gh(rng, N, 28 if wide else 16, dev)
        W = N - 1000
        wl = torch.from_numpy(rng.choice(N, W, replace=False).astype(np.int32)).to(dev)
        nb_seg = {"many_tiny": max(10, int(3000 * scale)), "few_huge": 3, "mixed": 200, "higgs_like": 50}[case]
        beg, end = segments(rng, W, nb_seg, dev)
        C = nb_seg + 5
        dest = torch.from_numpy(rng.permutation(C)[:nb_seg].astype(np.int32)).to(dev)
        fm = torch.from_numpy(rng.random((nb_seg, d)) < 0.75).to(dev)
        shape = (2, C, d, B) if wide else (C, d, B)
        o_r = torch.zeros(shape, dtype=torch.int64, device=dev)
        o_g = torch.zeros(shape, dtype=torch.int64, device=dev)
        KR.build_hist(X, xrow, gh, wl, beg, end, dest, fm, o_r, wide)
        KG.build_hist(X, xrow, gh, wl, beg, end, dest, fm, o_g, wide)
        sync()
        assert bits_equal(o_r, o_g), describe_mismatch(o_r, o_g)
        o_g2 = torch.zeros(shape, dtype=torch.int64, device=dev)
        KG.build_hist(X, xrow, gh, wl, beg, end, dest, fm, o_g2, wide)
        assert bits_equal(o_g, o_g2), "build_hist not deterministic across launches"
        return f"N={N} nb={nb_seg} d={d}"

    for wide in (False, True):
        for case in ("many_tiny", "few_huge", "mixed", "higgs_like"):
            run_check(sec, f"build_hist[wide={wide},{case}]", lambda w=wide, c=case: t_build_hist(w, c))

    def t_node_sums_leaf():
        N = int(200000 * scale)
        gh = rand_gh(rng, N, 28, dev)
        wl = torch.from_numpy(rng.permutation(N).astype(np.int32)).to(dev)
        beg, end = segments(rng, N, 777, dev)
        assert bits_equal(KR.node_sums(gh, wl, beg, end), KG.node_sums(gh, wl, beg, end)), "node_sums"
        pr = torch.randn(N, device=dev)
        pg = pr.clone()
        lv = torch.randn(777, device=dev)
        KR.leaf_update(pr, wl, beg, end, lv)
        KG.leaf_update(pg, wl, beg, end, lv)
        assert bits_equal(pr, pg), describe_mismatch(pr, pg)
    run_check(sec, "node_sums+leaf_update", t_node_sums_leaf)

    def t_partition(P):
        N, d, B = int(300000 * scale), 12, 256
        X = torch.from_numpy(rng.integers(0, B, (N, d)).astype(np.uint8)).to(dev)
        xrow = torch.from_numpy(rng.permutation(N).astype(np.int32)).to(dev)
        W = N - 5000
        wl = torch.from_numpy(rng.choice(N, W, replace=False).astype(np.int32)).to(dev)
        beg, end = segments(rng, W, P, dev, nonempty=True)
        P = len(beg)
        feat = torch.from_numpy(rng.integers(0, d, P).astype(np.int32)).to(dev)
        thr = torch.from_numpy(rng.integers(-1, B, P).astype(np.int32)).to(dev)
        wr, wg = wl.clone(), wl.clone()
        mr = KR.partition(X, xrow, wr, beg, end, feat, thr)
        mg = KG.partition(X, xrow, wg, beg, end, feat, thr)
        assert bits_equal(mr, mg), "mid: " + describe_mismatch(mr, mg)
        assert bits_equal(wr, wg), "wl (stability): " + describe_mismatch(wr, wg)
    for P in (1, 64, 4000):
        run_check(sec, f"partition[P={P}]", lambda p=P: t_partition(p))

    def t_split(B, d, C):
        G = torch.from_numpy(rng.integers(-3000, 3000, (C, d, B))).to(dev)
        H = torch.from_numpy(rng.integers(0, 5000, (C, d, B))).to(dev)
        H[:, :, B // 2:] = 0
        G[0] = 0
        H[0] = 0
        G[1, 1] = G[1, 0]
        H[1, 1] = H[1, 0]
        Gt, Ht = G.sum((1, 2)), H.sum((1, 2))
        isg = torch.rand(C, device=dev) * 1e-2
        ish = torch.rand(C, device=dev) * 1e-3
        lam = torch.rand(C, device=dev) + 0.1
        mcw = torch.rand(C, device=dev) * 2
        fm = torch.from_numpy(rng.random((C, d)) < 0.8).to(dev)
        r = KR.split_search(G, H, Gt, Ht, isg, ish, lam, mcw, fm)
        g = KG.split_search(G, H, Gt, Ht, isg, ish, lam, mcw, fm)
        for name, a, b in zip(("feat", "bin", "gain", "GL", "HL"), r, g):
            assert bits_equal(a, b), f"{name}: " + describe_mismatch(a, b)
    for B, d, C in ((16, 5, 300), (256, 28, max(8, int(2000 * SZ))), (256, 14, 7), (64, 23, max(8, int(1000 * SZ)))):
        run_check(sec, f"split_search[B={B},d={d},C={C}]", lambda b=B, dd=d, c=C: t_split(b, dd, c))

    def t_subtract_decode(wide):
        C, d, B = max(8, int(300 * SZ)), 28, 256
        h = rand_gh(rng, C * d * B, 16, dev).reshape(C, d, B)
        if not wide:
            assert bits_equal(KR.decode_hist(h, False), KG.decode_hist(h, False)), "decode"
            src = h
            o_r, o_g = torch.zeros_like(h), torch.zeros_like(h)
        else:
            src = torch.stack(KR.decode_hist(h, False))
            o_r, o_g = torch.zeros_like(src), torch.zeros_like(src)
        oi = torch.from_numpy(rng.permutation(C)[:C // 3]).to(dev)
        pi = torch.from_numpy(rng.permutation(C)[:C // 3]).to(dev)
        si = torch.from_numpy(rng.permutation(C)[:C // 3]).to(dev)
        KR.subtract_into(o_r, oi, src, pi, src, si, wide)
        KG.subtract_into(o_g, oi, src, pi, src, si, wide)
        assert bits_equal(o_r, o_g), describe_mismatch(o_r, o_g)
    for wide in (False, True):
        run_check(sec, f"subtract_into+decode[wide={wide}]", lambda w=wide: t_subtract_decode(w))

    def t_quantize(has_hess):
        RT, S = int(500000 * scale), 37
        slot = torch.from_numpy(np.sort(rng.integers(0, S, RT)).astype(np.int32)).to(dev)
        work = torch.rand(RT, device=dev) < 0.7
        g = torch.randn(RT, device=dev) * torch.from_numpy(rng.choice([1e-4, 1.0, 50.0], RT)).float().to(dev)
        h = torch.rand(RT, device=dev) * 0.25
        sel = slot == 5
        g[sel] = torch.from_numpy(rng.integers(-32000, 32000, int(sel.sum()))).float().to(dev) + 0.5
        g[torch.nonzero(sel)[0]] = 32767.0
        bits = torch.from_numpy(rng.choice([16, 28], S).astype(np.int8)).to(dev)
        r = KR.quantize(g, h, slot, work, bits, S, has_hess)
        q = KG.quantize(g, h, slot, work, bits, S, has_hess)
        for name, a, b in zip(("gh", "sg", "sh"), r, q):
            assert bits_equal(a, b), f"{name}: " + describe_mismatch(a, b)
        rt_off = torch.searchsorted(slot, torch.arange(S + 1, dtype=torch.int32, device=dev)).to(torch.int64)
        outs = {"ref[rt_off]": KR.quantize(g, h, slot, work, bits, S, has_hess, rt_off=rt_off)}
        mode0 = KG.QMAX
        try:
            for mode in ("scatter", "segment", "triton"):
                KG.QMAX = mode
                outs[f"gpu[{mode}]"] = KG.quantize(g, h, slot, work, bits, S, has_hess, rt_off=rt_off)
        finally:
            KG.QMAX = mode0
        for tag, q in outs.items():
            for name, a, b in zip(("gh", "sg", "sh"), r, q):
                assert bits_equal(a, b), f"{tag} {name}: " + describe_mismatch(a, b)
    for hh in (True, False):
        run_check(sec, f"quantize[has_hess={hh}]", lambda x=hh: t_quantize(x))

    def t_route():
        N, d, B = int(100000 * scale), 10, 64
        X = torch.from_numpy(rng.integers(0, B, (N, d)).astype(np.uint8)).to(dev)
        xrow = torch.from_numpy(rng.permutation(N).astype(np.int32)).to(dev)
        S, D = 40, 6
        levels, leaf = [], []
        for s in range(S):
            nodes = [0]
            for l in range(D):
                nxt = []
                for n in nodes:
                    if rng.random() < 0.75:
                        levels.append((l, (s << 13) | n, int(rng.integers(0, d)), int(rng.integers(-1, B))))
                        nxt += [2 * n + 1, 2 * n + 2]
                    else:
                        leaf.append((s << 13) | n)
                nodes = nxt
            leaf += [(s << 13) | n for n in nodes]
        lv = []
        for l in range(D):
            e = sorted([x for x in levels if x[0] == l], key=lambda x: x[1])
            lv.append((torch.tensor([x[1] for x in e], dtype=torch.int64, device=dev), torch.tensor([x[2] for x in e], dtype=torch.int32, device=dev),
                       torch.tensor([x[3] for x in e], dtype=torch.int32, device=dev)))
        leaf_key = torch.tensor(sorted(leaf), dtype=torch.int64, device=dev)
        leaf_val = torch.randn(len(leaf), device=dev)
        q = torch.from_numpy(np.sort(rng.choice(N, N * 3 // 4, replace=False))).to(dev)
        slot_q = torch.from_numpy(rng.integers(0, S, len(q))).to(dev)
        pr = torch.randn(N, device=dev)
        pg = pr.clone()
        KR.route_update(X, xrow, pr, q, slot_q, lv, leaf_key, leaf_val, 13)
        KG.route_update(X, xrow, pg, q, slot_q, lv, leaf_key, leaf_val, 13)
        assert bits_equal(pr, pg), describe_mismatch(pr, pg)
    run_check(sec, "route_update", t_route)


class Shadow:
    INPLACE = {"build_hist": -2, "subtract_into": 0, "partition": 2, "leaf_update": 0, "route_update": 2}

    def __init__(self, KG):
        self.KG = KG
        self.calls = {}
        self.mismatch = {}

    def __getattr__(self, name):
        ref = getattr(KR, name)
        if not callable(ref):
            return ref
        gpu = getattr(self.KG, name)
        inplace = self.INPLACE.get(name)

        def both(*args, **kw):
            self.calls[name] = self.calls.get(name, 0) + 1
            if inplace is not None:
                tgt = args[inplace]
                snap = tgt.clone()
                r = ref(*args, **kw)
                tr = tgt.clone()
                tgt.copy_(snap)
                try:
                    g = gpu(*args, **kw)
                    ok = bits_equal(tr, tgt) and (r is None or bits_equal(r, g))
                    info = "" if ok else (describe_mismatch(tr, tgt) if not bits_equal(tr, tgt) else describe_mismatch(r, g))
                except Exception as e:
                    ok, info = False, f"{type(e).__name__}: {e}"
                tgt.copy_(tr)
                if not ok and name not in self.mismatch:
                    self.mismatch[name] = f"call #{self.calls[name]} shapes={[tuple(a.shape) for a in args if torch.is_tensor(a)]}: {info}"
                return r
            r = ref(*args, **kw)
            try:
                g = gpu(*args, **kw)
                ok = bits_equal(r, g)
                info = "" if ok else describe_mismatch(r, g)
            except Exception as e:
                ok, info = False, f"{type(e).__name__}: {e}"
            if not ok and name not in self.mismatch:
                self.mismatch[name] = f"call #{self.calls[name]} shapes={[tuple(a.shape) for a in args if torch.is_tensor(a)]}: {info}"
            return r
        return both


class Counting:
    def __init__(self, mod):
        self.mod = mod
        self.calls = {}

    def __getattr__(self, name):
        a = getattr(self.mod, name)
        if not callable(a):
            return a

        def f(*args, **kw):
            self.calls[name] = self.calls.get(name, 0) + 1
            return a(*args, **kw)
        return f


class BackendEngine:
    def __init__(self, mod, cohorts, loss, **kw):
        self.mod = mod
        from mangrove import kernels_gpu as KG
        if mod is KG or mod is KR:
            self.eng = Engine(cohorts, loss, device=DEV, backend="gpu" if mod is KG else "ref", **kw)
            assert self.eng.KR is mod, "engine did not bind the requested backend module"
            return
        old = ENG.KR
        ENG.KR = mod
        try:
            self.eng = Engine(cohorts, loss, device=DEV, backend="ref", **kw)
        finally:
            ENG.KR = old

    def __getattr__(self, name):
        a = getattr(self.eng, name)
        if not callable(a):
            return a

        def f(*args, **kw):
            old = ENG.KR
            ENG.KR = self.mod
            try:
                return a(*args, **kw)
            finally:
                ENG.KR = old
        return f


def obs(eng, mid):
    st = eng.reg[mid]
    parts = [st.local_round, tree_sig(st.trees), tuple(st.metric_q_history), st.status, st.stop_reason, st.best_round]
    if len(st.spec.eval_rows):
        parts.append(eng.eval_margin(mid).view(np.int32).tobytes())
    return tuple(parts)


def compare_models(e_ref, e_gpu, ids):
    bad = []
    for mid in ids:
        a, b = obs(e_ref, mid), obs(e_gpu, mid)
        if a != b:
            st = e_ref.reg[mid]
            why = []
            if a[0] != b[0]:
                why.append(f"rounds {a[0]} vs {b[0]}")
            for t, (ta, tb) in enumerate(zip(st.trees, e_gpu.reg[mid].trees)):
                if tree_sig([ta]) != tree_sig([tb]):
                    why.append(f"first differing tree: round {t}")
                    break
            if a[2] != b[2]:
                why.append("metric numerators differ")
            if len(a) > 6 and a[-1] != b[-1]:
                why.append("eval margins differ")
            bad.append(f"model {mid} ({st.spec.tag}): {', '.join(why) or 'status/stop differ'}")
    return bad


def shadow_runs(dev, KG, full):
    sec = "shadow"
    cases = [("binary", int(12000 * min(1, SZ)) if not full else None, 6, 8), ("multiclass", int(6000 * min(1, SZ)) if not full else None, 3, 6),
             ("regression", int(8000 * min(1, SZ)) if not full else None, 3, 6)]
    if full:
        cases.append(("binary-wide", None, 4, 10))
    for name, n_max, ncfg, rounds in cases:
        def run(name=name, n_max=n_max, ncfg=ncfg, rounds=rounds):
            Xb, y, folds, loss, K = table(name, n_max=n_max, seed=1)
            co = Cohort("g", Xb, y)
            cfgs = sample_hpo(ncfg, seed=5, depth="mixed", dmin=2, dmax=8, sub=(0.5, 1.0), col=(0.5, 1.0), bits=(16, 28))
            specs = make_specs(cfgs, folds, n_rounds=rounds, fold_list=[0, 1])
            sh = Shadow(KG)
            eng = BackendEngine(sh, [co], loss, n_classes=K)
            eng.add_models(specs)
            eng.run()
            sync()
            RESULTS.setdefault("shadow_calls", {})[name] = sh.calls
            if sh.mismatch:
                raise AssertionError("; ".join(f"{k}: {v}" for k, v in sh.mismatch.items()))
            return f"{len(specs)} models x {rounds} rounds, calls={sh.calls}"
        run_check(sec, f"shadow[{name}]", run)


def training_runs(dev, KG, full):
    sec = "training"
    KGc = Counting(KG)
    n_adult = None if full else int(10000 * min(1, SZ))
    rounds = 25 if full else (10 if SZ >= 1 else 4)

    def workload(name, cohorts, loss, K, specs, script=None, **kw):
        def run():
            e_ref = BackendEngine(KR, cohorts, loss, n_classes=K, **kw)
            e_gpu = BackendEngine(KGc, cohorts, loss, n_classes=K, **kw)
            ids_r = e_ref.add_models(specs)
            ids_g = e_gpu.add_models(specs)
            assert ids_r == ids_g
            if script is None:
                e_ref.run()
                e_gpu.run()
            else:
                script(e_ref)
                script(e_gpu)
            sync()
            assert KGc.calls.get("build_hist", 0) > 0, "gpu backend was never called (engine does not read mangrove.engine.KR?)"
            bad = compare_models(e_ref.eng, e_gpu.eng, ids_r)
            assert not bad, "; ".join(bad[:5])
            assert e_ref.events == e_gpu.events, "event logs differ"
            return f"{len(specs)} models identical; events={len(e_ref.events)}"
        run_check(sec, name, run)

    Xb, y, folds, loss, K = table("binary", n_max=n_adult, seed=2)
    co = [Cohort("g", Xb, y)]
    cfgs = sample_hpo(8 if not full else 16, seed=11, depth="mixed", dmin=1, dmax=10, sub=(0.5, 1.0), col=(0.4, 1.0), bits=(16, 28))
    specs = make_specs(cfgs, folds, n_rounds=rounds, fold_list=[0, 1])
    workload("W1_logistic_mixed", co, loss, K, specs)
    W1 = (co, loss, K, specs)
    Xb2, y2, folds2, loss2, K2 = table("regression", n_max=None if full else int(8000 * min(1, SZ)), seed=2)
    cfgs2 = sample_hpo(6, seed=12, depth="mixed", dmin=2, dmax=8, sub=(0.6, 1.0), col=(0.5, 1.0), bits=(16, 28))
    workload("W2_squared", [Cohort("c", Xb2, y2)], loss2, K2, make_specs(cfgs2, folds2, data_key="c", n_rounds=rounds, fold_list=[0, 3]))
    Xb3, y3, folds3, loss3, K3 = table("multiclass", n_max=None if full else int(5000 * min(1, SZ)), seed=2)
    cfgs3 = sample_hpo(3, seed=13, depth="mixed", dmin=2, dmax=6, sub=(0.7, 1.0), col=(0.6, 1.0), bits=(16,))
    workload("W3_softmax", [Cohort("l", Xb3, y3)], loss3, K3, make_specs(cfgs3, folds3, data_key="l", n_rounds=max(4, rounds // 2), fold_list=[0]))
    coA = Cohort("A", Xb, y)
    coB = Cohort("B", (Xb.astype(np.int32) // 3).astype(np.uint8), y)
    rows = np.arange(len(y))
    sA = [ModelSpec("A", rows[folds != 0], rows[folds == 0], n_rounds=rounds, seed=300 + i, **c) for i, c in enumerate(cfgs[:4])]
    sB = [ModelSpec("B", rows[folds != 1], rows[folds == 1], n_rounds=rounds, seed=400 + i, **c) for i, c in enumerate(cfgs[:4])]
    workload("W4_two_cohorts", [coA, coB], loss, K, sA + sB)

    def script(e):
        rng = np.random.default_rng(0)
        step = 0
        while e.pending or e.active_ids():
            e.step()
            step += 1
            if step == 3 and e.active_ids():
                e.retire([e.active_ids()[0]])
            if step % 4 == 0 and len(e.active_ids()) > 1:
                e.compact(order=list(rng.permutation(e.active_ids())))
            if step > 400:
                raise RuntimeError("script did not terminate")
    specs5 = make_specs(cfgs[:6], folds, n_rounds=rounds // 2 + 2, fold_list=[0, 1])
    for i, s in enumerate(specs5):
        s.n_rounds = 3 + (i % 5)
    workload("W5_lifecycle_events", co, loss, K, specs5, script=script, max_model_slots=5)
    if full:
        for pause_loss, pause_K in (("logistic", 1), ("softmax", 3)):
            run_check(sec, f"W6_pause_resume_{pause_loss}",
                      lambda loss=pause_loss, K=pause_K: pause_resume_check(KGc, loss, K))
    return W1, KGc


def pause_resume_check(mod, loss, K):
    Xb, y, folds = synth(192, 5, seed=73, task="binary" if K == 1 else "multiclass", K=K)
    co = [Cohort("p", Xb, y)]
    cfgs = sample_hpo(3, seed=74, dmin=2, dmax=3, sub=(0.6, 0.9), col=(0.6, 0.9), bits=(28,))
    specs = make_specs(cfgs, folds, data_key="p", n_rounds=5, fold_list=[0, 1])
    engines = [BackendEngine(backend, co, loss, n_classes=K, max_model_slots=2)
               for backend in (KR, mod, mod)]
    ids = [engine.add_models(specs) for engine in engines]
    assert ids[0] == ids[1] == ids[2]
    engines[0].run()
    engines[1].run()
    scheduled = engines[2]
    scheduled.step()
    paused = scheduled.active_ids()[:1]
    assert paused, "pause check did not admit a model"
    prefix = [(scheduled.reg[mid].local_round, tree_sig(scheduled.reg[mid].trees),
               tuple(scheduled.reg[mid].metric_q_history)) for mid in paused]
    scheduled.pause(paused)
    scheduled.step()
    scheduled.step()
    assert prefix == [(scheduled.reg[mid].local_round, tree_sig(scheduled.reg[mid].trees),
                       tuple(scheduled.reg[mid].metric_q_history)) for mid in paused], "paused model advanced"
    scheduled.compact(order=list(reversed(scheduled.active_ids())))
    scheduled.resume(paused, admit=False)
    scheduled.run()
    sync()
    for reference in engines[:2]:
        bad = compare_models(reference.eng, scheduled.eng, ids[0])
        assert not bad, "; ".join(bad[:5])
    return f"{len(specs)} models; K={K}; paused prefix and final trees/margins/metric numerators identical"


def fused_vs_standalone(dev, KG, W1, full):
    sec = "fused"
    co, loss, K, specs = W1
    sub = specs if full else specs[:6]

    def run():
        fused = BackendEngine(KG, co, loss, n_classes=K)
        ids = fused.add_models(sub)
        fused.run()
        bad = []
        for sp, mid in zip(sub, ids):
            solo = BackendEngine(KG, co, loss, n_classes=K)
            sid = solo.add_models([sp])[0]
            solo.run()
            if obs(fused.eng, mid) != obs(solo.eng, sid):
                bad.append(mid)
        sync()
        assert not bad, f"models differ fused vs standalone: {bad}"
        return f"{len(sub)} models identical"
    run_check(sec, "fused_vs_one_at_a_time_gpu", run)


def main():
    global SZ
    full = True
    mode = "full"
    SZ = 4.0
    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "check_full.json"
    if DEV == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available"); sys.exit(2)
        dev = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(0.75, dev)
    else:
        dev = torch.device("cpu")
    torch.manual_seed(0)
    RESULTS["mode"] = mode
    RESULTS["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        env_report(dev)
        from mangrove import kernels_gpu as KG
    except Exception as e:
        record("env", "load", False, f"{type(e).__name__}: {e}")
        finish(out, 1)
        return
    record("env", "load", True, "kernel library + triton loaded")
    random_kernel_checks(dev, KG, full)
    shadow_runs(dev, KG, full)
    W1, KGc = training_runs(dev, KG, full)
    RESULTS["gpu_calls_training"] = KGc.calls
    fused_vs_standalone(dev, KG, W1, full)
    finish(out, 1 if RESULTS["failures"] else 0)


def finish(out, code):
    RESULTS["seconds"] = round(time.time() - T0, 1)
    RESULTS["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if DEV == "cuda" else None
    RESULTS["exit_code"] = code
    out.write_text(json.dumps(RESULTS, indent=1, default=str))
    lines = [f"check {RESULTS.get('mode')}  {RESULTS['started']}  {RESULTS['seconds']}s  exit={code}"]
    env = RESULTS.get("env", {})
    if env:
        lines.append(f"device {env.get('device')} torch {env.get('torch')} cuda {env.get('cuda')} triton {env.get('triton')} lib v{env.get('kernel_lib', {}).get('version')}")
    for sec, rows in RESULTS["sections"].items():
        for r in rows:
            lines.append(f"{r['status']:4s} {sec:10s} {r['name']:45s} {r['seconds']:8.2f}s  {r['info'].splitlines()[0] if r['info'] else ''}")
    lines.append(f"{len(RESULTS['failures'])} failure(s)")
    txt = "\n".join(lines)
    out.with_suffix(".txt").write_text(txt)
    print("\n" + txt)
    print(f"\nwrote {out} and {out.with_suffix('.txt')}")
    sys.exit(code)


if __name__ == "__main__":
    main()
