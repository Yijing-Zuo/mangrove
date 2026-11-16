import numpy as np
import pytest
import torch

from util import obs, synth
from mangrove import Engine, Cohort, ModelSpec
import mangrove.engine as engine_module

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _nm_library_state():
    from mangrove import kernels_nm
    try:
        kernels_nm.lib(torch.device("cuda"))
    except AttributeError as e:
        return f"kernel library without node-map entry points ({e})"
    return "ok"


if torch.cuda.is_available():
    _state = _nm_library_state()
    if _state != "ok":
        pytestmark = pytest.mark.skip(reason=_state)


def _cv_specs(n, folds, rng, M, depth=(3, 7), sub=None, col=None, rounds=4, K=1):
    specs = []
    for i in range(M):
        k = i % folds
        rows = np.arange(n)
        tr, ev = rows[rows % folds != k], rows[rows % folds == k]
        d = int(rng.integers(depth[0], depth[1] + 1))
        specs.append(ModelSpec("c", tr, ev, eta=float(rng.uniform(0.05, 0.3)), lam=float(rng.uniform(0.1, 5)),
                               mcw=float(rng.uniform(0.0, 3.0)), max_depth=d, n_rounds=rounds,
                               subsample=float(rng.uniform(*sub)) if sub else 1.0,
                               colsample_bytree=float(rng.uniform(*col)) if col else 1.0, seed=100 + i))
    return specs


def _pair(co, loss, specs, K=1, **kw):
    out = []
    for nm in (False, True):
        e = Engine(co, loss, n_classes=K, device="cuda", backend="gpu", nodemap=nm, **kw)
        ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs])
        e.run()
        out.append((e, ids))
    return out


def _assert_same(pair, need_nm=True):
    (a, ia), (b, ib) = pair
    bad = [i for i, (x, z) in enumerate(zip(ia, ib)) if obs(a, x) != obs(b, z)]
    assert not bad, f"models differ: {bad[:10]}"
    assert a.nm_stats["rounds"] == 0
    if need_nm:
        assert b.nm_stats["rounds"] > 0, b.nm_stats


@pytest.mark.parametrize("loss,K", [("logistic", 1), ("squared", 1), ("softmax", 4)])
def test_losses_sampling_mixed_depth(loss, K):
    rng = np.random.default_rng(7)
    task = {"logistic": "binary", "squared": "regression", "softmax": "multiclass"}[loss]
    Xb, y, _ = synth(3000, 9, nb=40, seed=3, task=task, K=K)
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(3000, 4, rng, 12, sub=(0.6, 1.0), col=(0.5, 1.0))
    _assert_same(_pair(co, loss, specs, K=K))


def test_wide_mode_and_byte_loads():
    rng = np.random.default_rng(11)
    Xb, y, _ = synth(90000, 7, nb=64, seed=5, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(90000, 5, rng, 5, depth=(5, 6), rounds=3)
    assert all(s.resolve_mode() == "wide" for s in specs)
    _assert_same(_pair(co, "logistic", specs))


@pytest.mark.parametrize("deep_mn", [2, 128])
def test_deep_threshold_invariance(monkeypatch, deep_mn):
    monkeypatch.setattr(engine_module, "NM_DEEP_MN", deep_mn)
    rng = np.random.default_rng(13)
    Xb, y, _ = synth(4000, 8, nb=32, seed=9, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(4000, 4, rng, 10, depth=(6, 7), sub=(0.7, 1.0))
    pair = _pair(co, "logistic", specs)
    _assert_same(pair)
    st = pair[1][0].nm_stats
    if deep_mn == 2:
        assert st["deep_calls"] > 0
    else:
        assert st["deep_calls"] == 0


def test_launch_shape_invariance(monkeypatch):
    from mangrove import kernels_nm as KN
    rng = np.random.default_rng(17)
    Xb, y, _ = synth(5000, 12, nb=48, seed=2, task="regression")
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(5000, 5, rng, 10, depth=(4, 7), col=(0.6, 1.0))
    base = _pair(co, "squared", specs)
    _assert_same(base)
    monkeypatch.setattr(KN, "FORCE_RCH", 2048)
    monkeypatch.setattr(KN, "TABLE_KB", 4)
    monkeypatch.setattr(KN, "TABLE_KB_ROOT", 4)
    monkeypatch.setattr(KN, "NM_THREADS", 128)
    monkeypatch.setattr(KN, "HIST_THREADS", 128)
    monkeypatch.setattr(engine_module, "NM_DEEP_FG1", False)
    e = Engine(co, "squared", device="cuda", backend="gpu", nodemap=True)
    ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs])
    e.run()
    a, ia = base[0]
    assert all(obs(a, x) == obs(e, z) for x, z in zip(ia, ids))
    assert e.nm_stats["rounds"] > 0


def test_forced_chunking():
    rng = np.random.default_rng(19)
    Xb, y, _ = synth(3000, 6, nb=24, seed=4, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(3000, 3, rng, 9, depth=(5, 7))
    pair = _pair(co, "logistic", specs, chunk_parents=1)
    _assert_same(pair)
    e = pair[1][0]
    assert any(c["chunk"] < c["n_groups"] for c in e.counters)
    n_levels_split = sum(1 for c in e.counters if c["n_split"] > 0)
    assert e.nm_stats["update_calls"] == n_levels_split, (e.nm_stats, n_levels_split)


@pytest.mark.parametrize("cs", [1.0, 0.6])
def test_level_failure_after_partial_growth_retries_exactly(monkeypatch, cs):
    from mangrove.memory import LevelInfeasible
    rng = np.random.default_rng(834)
    Xb = rng.integers(0, 16, (400, 4), dtype=np.uint8)
    yy = (rng.standard_normal(400) + Xb[:, 0] * 0.3 - Xb[:, 1] * 0.1).astype(np.float32)
    rows = np.arange(400)
    specs = [ModelSpec("g", rows[rows % 5 != i], rows[rows % 5 == i], n_rounds=4, max_depth=5, mcw=0.0,
                       seed=91 + i, subsample=0.8, colsample_bytree=cs, acc_mode="wide") for i in range(4)]
    co = [Cohort("g", Xb, yy)]
    a = Engine(co, "squared", device="cuda", backend="gpu", nodemap=False)
    ia = a.add_models(specs); a.run()
    b = Engine(co, "squared", device="cuda", backend="gpu", nodemap=True)
    ib = b.add_models(specs)
    original = b._plan_core
    injected = []

    def plan(chunk, ngrp, level, has_parent, fixed_bytes, request, rows_):
        if b.step_index == 1 and level == 2 and not injected:
            injected.append(True)
            raise LevelInfeasible("test failure after partial node-map growth", 1001, 1000)
        return original(chunk, ngrp, level, has_parent, fixed_bytes, request, rows_)

    monkeypatch.setattr(b, "_plan_core", plan)
    o_round, o_degrade = b._round, b._degrade
    at_start, seen = [], []

    def round_():
        at_start.append(dict(b.nm_stats))
        return o_round()

    def degrade(*args, **kw):
        seen.append(dict(b.nm_stats) == at_start[-1])
        return o_degrade(*args, **kw)

    monkeypatch.setattr(b, "_round", round_)
    monkeypatch.setattr(b, "_degrade", degrade)
    b.run()
    assert injected and b.n_evictions == 1
    assert seen == [True], (seen, at_start[-2:], b.nm_stats)
    assert all(obs(a, x) == obs(b, z) for x, z in zip(ia, ib))
    assert b.nm_stats["rounds"] > 0
    if cs < 1.0:
        assert b.nm_stats["fused_colmask"] > 0 and not b._nm_ring["open"], b.nm_stats
        t = b._nm_fm_tree[0]
        r = b._nm_ring
        assert not (r["d"].data_ptr() <= t.data_ptr() < r["d"].data_ptr() + r["cap"])


def test_lifecycle_events():
    rng = np.random.default_rng(23)
    Xb, y, _ = synth(2500, 6, nb=20, seed=8, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(2500, 5, rng, 10, depth=(3, 7), sub=(0.7, 1.0), rounds=6)
    res = []
    for nm in (False, True):
        e = Engine(co, "logistic", device="cuda", backend="gpu", nodemap=nm, max_model_slots=4)
        ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs[:7]])
        e.step(); e.step()
        e.pause([ids[1]])
        e.retire([ids[2]])
        e.step()
        e.resume([ids[1]])
        e.compact()
        ids += e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs[7:]])
        e.run()
        res.append((e, ids))
    _assert_same(res)


def test_two_cohorts_subsets_and_waves():
    rng = np.random.default_rng(29)
    XA, yA, _ = synth(2600, 8, nb=24, seed=21, task="binary")
    XB, yB, _ = synth(1900, 8, nb=24, seed=22, task="binary")
    co = [Cohort("A", XA, yA), Cohort("B", XB, yB)]
    specs = []
    for i in range(14):
        key, n = ("A", 2600) if i % 2 == 0 else ("B", 1900)
        rows = np.arange(n)
        k = i % 4
        tr, ev = rows[rows % 4 != k], rows[rows % 4 == k]
        if i % 3 == 1:
            tr = tr[rng.random(len(tr)) < 0.75]
        specs.append(ModelSpec(key, tr, ev, eta=0.2, lam=1.0, mcw=0.5, max_depth=int(rng.integers(3, 8)),
                               n_rounds=int(rng.integers(2, 7)), seed=300 + i))
    seen = []

    def record(eng):
        xb = eng._nm_xbase()
        seen.extend(int(xb[s]) for s in range(eng.S) if eng.mslots[s // eng.K] is not None)

    res = []
    for nm in (False, True):
        e = Engine(co, "logistic", device="cuda", backend="gpu", nodemap=nm, max_model_slots=3)
        ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs])
        e.run(callback=record if nm else None)
        if nm:
            e.nm_check()
        res.append((e, ids))
    _assert_same(res)
    seen = np.asarray(seen)
    assert (seen == -1).any() and (seen == 0).any() and (seen > 0).any(), sorted(set(seen.tolist()))


def test_node_map_charged_only_where_used():
    Xb, y, _ = synth(3000, 6, nb=16, seed=31, task="binary")
    co = [Cohort("c", Xb, y)]
    rows = np.arange(3000)

    def need(nm, d, bits):
        e = Engine(co, "logistic", device="cuda", backend="gpu", nodemap=nm)
        mid = e.add_models([ModelSpec("c", rows[rows % 3 != 0], rows[rows % 3 == 0], max_depth=d, n_rounds=2,
                                      quant_bits=bits, seed=5)])[0]
        t = e._terms_total([(e.reg[mid], None)])
        return e._transient_of(t), e.planner.nm_bytes_per_rt

    for d, bits in ((9, 16), (6, 28)):
        (a, ca), (b, cb) = need(False, d, bits), need(True, d, bits)
        assert a == b and ca == cb == 0, (d, bits, a, b, ca, cb)
    (a, ca), (b, cb) = need(False, 6, 16), need(True, 6, 16)
    assert b > a and ca == 0 and cb == 1, (a, b, ca, cb)


def test_early_leaves_and_breaks():
    rng = np.random.default_rng(41)
    Xb, y, _ = synth(900, 5, nb=12, seed=12, task="binary")
    co = [Cohort("c", Xb, y)]
    rows = np.arange(900)
    specs = []
    for i in range(12):
        k = i % 4
        tr, ev = rows[rows % 4 != k], rows[rows % 4 == k]
        if i % 3 == 2:
            tr = tr[rng.random(len(tr)) < 0.08]
        mcw = (0.0, 2.0, 1e6)[i % 3]
        specs.append(ModelSpec("c", tr, ev, eta=0.2, lam=1.0, mcw=mcw, max_depth=1 + i % 7,
                               n_rounds=6 if mcw == 1e6 else 4, subsample=0.8 if i % 2 else 1.0, seed=500 + i))
    pair = _pair(co, "logistic", specs)
    _assert_same(pair)
    c = pair[1][0].counters
    assert any(r["n_cand"] < r["n_frontier"] for r in c), "no level with leaves at its head"
    assert any(r["n_split"] == 0 and r["n_cand"] > 0 for r in c), "no level that ended with n_new_par == 0"
    assert any(r["n_split"] > 0 and r["level"] == 6 for r in c), "no model reached the last level (for-else)"


def test_head_break_after_root_split():
    Xb, y, _ = synth(900, 5, nb=12, seed=12, task="binary")
    co = [Cohort("c", Xb, y)]
    rows = np.arange(900)
    tr, ev = rows[rows % 4 != 0], rows[rows % 4 == 0][:50]
    pos0, pos1 = tr[y[tr] == 0][:1], tr[y[tr] == 1][:1]
    specs = [ModelSpec("c", np.concatenate([pos0, pos1]), ev, eta=0.3, lam=1.0, mcw=0.0, max_depth=7, n_rounds=5,
                       seed=600 + j) for j in range(2)]
    specs.append(ModelSpec("c", tr, ev, eta=0.2, lam=1.0, mcw=1.0, max_depth=5, n_rounds=2, seed=610))
    pair = _pair(co, "logistic", specs)
    _assert_same(pair)
    c = pair[1][0].counters
    early = [r for r in c if r["step"] < 2]
    late = [r for r in c if r["step"] >= 2]
    assert early and max(r["level"] for r in early) > 0, "the ordinary model must grow past the root while mixed"
    assert late and max(r["level"] for r in late) == 0 and all(r["n_split"] > 0 for r in late), \
        "the 2-row models must split the root and stop at the level-1 head (ncand == 0 break at l > 0)"


@pytest.mark.parametrize("chunk", [1, 3])
def test_chunked_mixed_modes(chunk):
    rng = np.random.default_rng(47)
    Xb, y, _ = synth(3000, 6, nb=32, seed=14, task="binary")
    co = [Cohort("c", Xb, y)]
    rows = np.arange(3000)
    specs = []
    for i in range(6):
        k = i % 3
        specs.append(ModelSpec("c", rows[rows % 3 != k], rows[rows % 3 == k], eta=0.2, lam=1.0, mcw=1.0,
                               max_depth=int(rng.integers(4, 7)), n_rounds=3, subsample=0.9 if i % 3 == 1 else 1.0,
                               colsample_bytree=0.8 if i % 3 == 2 else 1.0, seed=700 + i,
                               acc_mode="wide" if i % 2 == 0 else "auto"))
    assert {s.resolve_mode() for s in specs} == {"wide", "packed"}
    pair = _pair(co, "logistic", specs, chunk_parents=chunk)
    _assert_same(pair)
    e = pair[1][0]
    assert any(c["chunk"] < c["n_groups"] for c in e.counters)
    n_levels_split = sum(1 for c in e.counters if c["n_split"] > 0)
    assert e.nm_stats["update_calls"] == n_levels_split, (e.nm_stats, n_levels_split)


def test_split_params_host_matches_device():
    rng = np.random.default_rng(5)
    g = np.concatenate([np.exp(rng.uniform(-80, 80, 300000)), [1.0, 32767.0, 65535.0, 134217727.0, 268435455.0,
                                                               1e-300, 1e300, 3.0, 7.0, 1.0 / 3.0]])
    sg16 = (np.float32(32767.0) / g.astype(np.float32)).astype(np.float64)
    sg28 = (np.float64(1.0) / g) * np.float64(134217727.0)
    for gm in (g, sg16, sg28, 65535.0 / g):
        t = torch.from_numpy(np.ascontiguousarray(gm)).cuda()
        dev = (1.0 / t).to(torch.float32).cpu().numpy()
        host = (np.float64(1.0) / gm).astype(np.float32)
        assert np.array_equal(dev.view(np.int32), host.view(np.int32))
    lam = np.asarray([0.0, 0.1, 1.0, 3.0, 1e-7, 123.456789, 1e30, 0.3, 2.0 / 3.0], dtype=np.float64)
    tl = torch.tensor(lam.tolist(), dtype=torch.float64).cuda()
    assert np.array_equal(tl.to(torch.float32).cpu().numpy().view(np.int32), lam.astype(np.float32).view(np.int32))


def test_split_params_ftz_fallback():
    Xb, y, _ = synth(400, 4, nb=8, seed=3, task="binary")
    e = Engine([Cohort("c", Xb, y)], "logistic", device="cuda", backend="gpu", nodemap=True)
    e.add_models([ModelSpec("c", np.arange(300), np.arange(300, 400), max_depth=3, n_rounds=3, seed=1)])
    e.step()
    S = e.S
    sg_h = np.full(S, 3e38, dtype=np.float64)
    sh_h = np.full(S, 1.0, dtype=np.float64)
    sg, sh = torch.from_numpy(sg_h).cuda(), torch.from_numpy(sh_h).cuda()
    ref = e._nm_sp4_host(sg_h, sh_h, sg, sh)
    assert ref[0].view(np.int32)[0] == 2378746
    if not torch.set_flush_denormal(True):
        pytest.skip("flush-denormal not supported on this CPU")
    try:
        got = e._nm_sp4_host(sg_h, sh_h, sg, sh)
    finally:
        torch.set_flush_denormal(False)
    assert e.nm_stats.get("sp4_device", 0) == 1
    assert got.dtype == np.float32 and got.tobytes() == ref.tobytes()


def _worklist_block_reduction(og, of, ob, ogl, ohl):
    if og.shape[1] == 1:
        return of[:, 0].contiguous(), ob[:, 0].contiguous(), og[:, 0].contiguous(), ogl[:, 0].contiguous(), ohl[:, 0].contiguous()
    best = og.max(dim=1).values
    j = (og == best[:, None]).to(torch.int8).argmax(dim=1)
    j = j[:, None]
    return (of.gather(1, j).squeeze(1), ob.gather(1, j).squeeze(1), og.gather(1, j).squeeze(1),
            ogl.gather(1, j).squeeze(1), ohl.gather(1, j).squeeze(1))


def _arena_planes(pk, C_chunk, pos0, C):
    sl = slice(pos0, pos0 + C)
    GL = pk[:2 * C_chunk].view(np.int64)[sl]
    HL = pk[2 * C_chunk:4 * C_chunk].view(np.int64)[sl]
    bg = pk[4 * C_chunk:5 * C_chunk][sl]
    bf = pk[5 * C_chunk:6 * C_chunk][sl]
    bb = pk[6 * C_chunk:7 * C_chunk][sl]
    return bf, bb, bg, GL, HL


def _assert_arena_equals(pk, C_chunk, pos0, C, ref):
    from mangrove.kernels_ref import KRTEPS
    bf, bb, bg, GL, HL = _arena_planes(pk, C_chunk, pos0, C)
    rf, rb, rg, rGL, rHL = (t.cpu().numpy() for t in ref)
    assert rf.dtype == np.int32 and rb.dtype == np.int32 and rg.dtype == np.float32 and rGL.dtype == np.int64
    assert np.array_equal(bf, rf), "best feature"
    assert np.array_equal(bb, rb), "best bin"
    assert np.array_equal(bg, rg.view(np.int32)), "gain bits (NaN payloads / signed zeros included)"
    assert np.array_equal(GL, rGL), "GL"
    assert np.array_equal(HL, rHL), "HL"
    assert np.array_equal(bg.view(np.float32) > np.float32(KRTEPS), (ref[2] > KRTEPS).cpu().numpy()), "split flags"


@pytest.mark.parametrize("NFB", [1, 2, 4, 12])
def test_split_finish_matches_torch(NFB):
    from mangrove import kernels_nm as KN
    rng = np.random.default_rng(1000 + NFB)
    dev = torch.device("cuda")
    C, pos0 = 517, 23
    C_chunk = pos0 + C + 41
    og = (rng.standard_normal((C, NFB)) * 1e-6).astype(np.float32)
    og[rng.random((C, NFB)) < 0.3] = -np.inf
    rows = rng.permutation(C)
    for r in rows[:80]:
        k = int(rng.integers(2, NFB + 1)) if NFB > 1 else 1
        cols = rng.choice(NFB, k, replace=False)
        og[r, cols] = og[r].max() if np.isfinite(og[r]).any() else np.float32(0.5)
    og[rows[80:120]] = -np.inf
    og[rows[120:130]] = np.inf
    for i, r in enumerate(rows[130:210]):
        og[r, 0 if (i % 2 == 0 or NFB == 1) else int(rng.integers(1, NFB))] = np.nan
    og[rows[210:230]] = 0.0
    og[rows[210:220], 0] = -0.0
    if NFB > 1:
        og[rows[220:230], 1] = -0.0
    of = rng.integers(0, 90, (C, NFB)).astype(np.int32)
    ob = rng.integers(0, 256, (C, NFB)).astype(np.int32)
    ogl = rng.integers(-(1 << 46), 1 << 46, (C, NFB)).astype(np.int64)
    ohl = rng.integers(0, 1 << 46, (C, NFB)).astype(np.int64)
    og_d, of_d, ob_d, ogl_d, ohl_d = (torch.from_numpy(a).to(dev) for a in (og, of, ob, ogl, ohl))
    RB = torch.full((7 * C_chunk + 64,), -7, dtype=torch.int32, device=dev)
    KN.split_finish(og_d, of_d, ob_d, ogl_d, ohl_d, RB, C_chunk, pos0)
    pk = RB.cpu().numpy()
    ref = _worklist_block_reduction(og_d, of_d, ob_d, ogl_d, ohl_d)
    _assert_arena_equals(pk[:7 * C_chunk], C_chunk, pos0, C, ref)
    nan_rows = np.flatnonzero(np.isnan(og).any(1))
    assert len(nan_rows) == 80
    bf, bb, bg, GL, HL = _arena_planes(pk[:7 * C_chunk], C_chunk, pos0, C)
    assert np.array_equal(bf[nan_rows], of[nan_rows, 0]) and np.array_equal(GL[nan_rows], ogl[nan_rows, 0])
    assert np.array_equal(bg[nan_rows], og[nan_rows, 0].view(np.int32))
    touched = np.zeros(7 * C_chunk + 64, dtype=bool)
    for o, w in ((0, 2), (2 * C_chunk, 2), (4 * C_chunk, 1), (5 * C_chunk, 1), (6 * C_chunk, 1)):
        touched[o + w * pos0:o + w * (pos0 + C)] = True
    assert (pk[~touched] == -7).all()


@pytest.mark.parametrize("d,B", [(16, 16), (20, 16), (64, 16), (8, 256), (90, 256)])
def test_split_finish_end_to_end_vs_split_search(d, B):
    from mangrove import kernels_gpu as KG, kernels_nm as KN
    rng = np.random.default_rng(d * 1000 + B)
    dev = torch.device("cuda")
    C = 61
    G = torch.from_numpy(rng.integers(-5000, 5000, (C, d, B)).astype(np.int64)).to(dev)
    H = torch.from_numpy(rng.integers(0, 3000, (C, d, B)).astype(np.int64)).to(dev)
    H[:, :, B // 2:] *= (torch.rand(C, d, B - B // 2, device=dev) < 0.7).to(torch.int64)
    G[torch.from_numpy(rng.integers(0, C, 7)).to(dev)] = 0
    Gt, Ht = G[:, 0].sum(1), H[:, 0].sum(1)
    isg = torch.from_numpy((1.0 / rng.uniform(1e3, 3e4, C)).astype(np.float32)).to(dev)
    ish = torch.from_numpy((1.0 / rng.uniform(1e3, 6e4, C)).astype(np.float32)).to(dev)
    lam = torch.from_numpy(rng.choice([0.0, 0.1, 1.0, 3.0], C).astype(np.float32)).to(dev)
    mcw = torch.from_numpy(rng.choice([0.0, 0.5, 1.0, 1e6], C).astype(np.float32)).to(dev)
    fmask = torch.from_numpy(rng.random((C, d)) < 0.8).to(dev)
    fmask[torch.from_numpy(rng.integers(0, C, 5)).to(dev)] = False
    ref = KG.split_search(G, H, Gt, Ht, isg, ish, lam, mcw, fmask)
    pos0, C_chunk, c0 = 9, 9 + C + 5, 25
    RB = torch.full((7 * C_chunk,), -7, dtype=torch.int32, device=dev)
    KN.split_search_nm(G[:c0], H[:c0], Gt[:c0], Ht[:c0], isg[:c0], ish[:c0], lam[:c0], mcw[:c0], fmask[:c0],
                       RB, C_chunk, pos0)
    KN.split_search_nm(G[c0:], H[c0:], Gt[c0:], Ht[c0:], isg[c0:], ish[c0:], lam[c0:], mcw[c0:], fmask[c0:],
                       RB, C_chunk, pos0 + c0)
    pk = RB.cpu().numpy()
    _assert_arena_equals(pk, C_chunk, pos0, C, ref)
    bg = _arena_planes(pk, C_chunk, pos0, C)[2].view(np.float32)
    assert np.isneginf(bg).any() and (bg > 1e-6).any(), "the case mix must include -inf rows and real splits"


def _torch_scale_chain(gmax, hmax, bits, has_hess):
    from mangrove import kernels_gpu as KG
    from mangrove.kernels_ref import G16, H16, G28, H28
    dev = gmax.device
    S = gmax.numel()
    one32 = torch.ones_like(gmax)
    sg32 = torch.where(gmax > 0, KG._dconst(G16, torch.float32, dev) / gmax, one32)
    g64 = gmax.to(torch.float64)
    sg28 = torch.where(g64 > 0, G28 / g64, torch.ones_like(g64))
    is28 = (bits == 28)
    sg = torch.where(is28, sg28, sg32.to(torch.float64))
    if has_hess:
        sh32 = torch.where(hmax > 0, KG._dconst(H16, torch.float32, dev) / hmax, one32)
        h64 = hmax.to(torch.float64)
        sh28 = torch.where(h64 > 0, H28 / h64, torch.ones_like(h64))
        sh = torch.where(is28, sh28, sh32.to(torch.float64))
    else:
        sh = torch.ones(S, dtype=torch.float64, device=dev)
        sh32, sh28 = one32, torch.ones_like(g64)
    return dict(sg32=sg32, sh32=sh32, sg28=sg28, sh28=sh28, is28=is28.view(torch.uint8), sg=sg, sh=sh)


def _scale_inputs(rng, n):
    with np.errstate(over="ignore"):
        v = np.exp(rng.uniform(-110, 90, n)).astype(np.float32)
    k = rng.permutation(n)
    v[k[:n // 20]] = 0.0
    v[k[n // 20:n // 10]] = rng.integers(1, 2 ** 23, n // 10 - n // 20).astype(np.int32).view(np.float32)
    v[k[n // 10:n // 10 + 500]] = np.ldexp(np.float32(1.0), rng.integers(-149, 128, 500)).astype(np.float32)
    v[k[n // 10 + 500:n // 10 + 1000]] = np.float32(np.finfo(np.float32).max) * rng.choice([1.0, 0.5, 0.999999], 500)
    v[k[n // 10 + 1000:n // 10 + 1100]] = np.inf
    v[k[n // 10 + 1100:n // 10 + 1200]] = np.nan
    v[k[n // 10 + 1200:n // 10 + 1300]] = np.float32(1.401298464324817e-45)
    v[k[n // 10 + 1300:n // 10 + 1310]] = [1.0, 32767.0, 65535.0, 134217727.0, 268435455.0, 3.0, 7.0, 1.0 / 3.0,
                                           2.0 ** -126, 2.0 ** 127]
    return v


@pytest.mark.parametrize("has_hess", [True, False])
def test_scales_match_torch_chain(has_hess):
    from mangrove import kernels_nm as KN
    rng = np.random.default_rng(404 + int(has_hess))
    dev = torch.device("cuda")
    n, s0 = 1_000_000, 37
    gmax = torch.from_numpy(_scale_inputs(rng, n)).to(dev)
    hmax = torch.from_numpy(_scale_inputs(np.random.default_rng(99), n)).to(dev)
    bits = torch.from_numpy(rng.choice([16, 28], n).astype(np.int8)).to(dev)
    ref = _torch_scale_chain(gmax, hmax if has_hess else None, bits, has_hess)
    scal = KN.Scal(s0 + n + 11, dev)
    scal.buf.fill_(-6510615555426900571)
    KN.scales(gmax, hmax if has_hess else None, bits, scal.views(s0, s0 + n), has_hess)
    for name, r in ref.items():
        got = getattr(scal, name)[s0:s0 + n]
        assert got.dtype == r.dtype, name
        gi, ri = got.contiguous().view(torch.uint8).cpu().numpy(), r.contiguous().view(torch.uint8).cpu().numpy()
        bad = np.flatnonzero(gi != ri)
        assert len(bad) == 0, (name, len(bad), gmax[bad[:5] // r.element_size()].tolist(), got[bad[:5] // r.element_size()].tolist(), r[bad[:5] // r.element_size()].tolist())
    for name, dt in KN.Scal.SECTIONS[:5] + KN.Scal.SECTIONS[8:]:
        t = getattr(scal, name).view(torch.uint8) if dt == torch.uint8 else getattr(scal, name).contiguous().view(torch.uint8)
        u = t.cpu().numpy().reshape(scal.S, -1)
        assert (u[:s0] == 0xA5).all() and (u[s0 + n:] == 0xA5).all(), name
    gm = gmax.cpu().numpy()
    assert (gm == 0).any() and np.isinf(gm).any() and np.isnan(gm).any()
    assert ((gm > 0) & (gm < np.finfo(np.float32).tiny)).sum() > 1000
    assert (ref["sg32"] != 1).sum() > n // 2 and (ref["sg"] != ref["sg32"].to(torch.float64)).sum() > n // 4


@pytest.mark.parametrize("qmax", ["segment", "triton", "scatter"])
def test_quantize_nm_matches_worklist_path(monkeypatch, qmax):
    from mangrove import kernels_gpu as KG, kernels_nm as KN
    monkeypatch.setattr(KG, "QMAX", qmax)
    rng = np.random.default_rng(2024)
    dev = torch.device("cuda")
    lens = np.array([5000, 0, 70000, 1, 333, 40000, 2])
    S = len(lens)
    rt_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    RT = int(rt_off[-1])
    slot = np.repeat(np.arange(S, dtype=np.int32), lens)
    work = rng.random(RT) < 0.7
    work[rt_off[3]:rt_off[4]] = False
    work[rt_off[6]:rt_off[7]] = True
    g = (rng.standard_normal(RT) * np.exp(rng.uniform(-30, 30, RT))).astype(np.float32)
    h = np.abs(rng.standard_normal(RT) * np.exp(rng.uniform(-30, 10, RT))).astype(np.float32)
    g[rng.random(RT) < 0.01] = 0.0
    bits = rng.choice([16, 28], S).astype(np.int8)
    t = lambda a, dt=None: torch.from_numpy(a).to(dev) if dt is None else torch.from_numpy(a).to(dev).to(dt)
    g_d, h_d, sl_d, w_d, bits_d, ro_d = t(g), t(h), t(slot), t(work), t(bits), t(rt_off)
    for has_hess in (True, False):
        gh0, sg0, sh0 = KG.quantize(g_d, h_d, sl_d, w_d, bits_d, S, has_hess=has_hess, rt_off=ro_d)
        s0 = 5
        scal = KN.Scal(s0 + S + 3, dev)
        gh1, sg1, sh1 = KN.quantize_nm(KG, g_d, h_d, sl_d, w_d, bits_d, S, has_hess, ro_d, scal, s0)
        assert torch.equal(gh0, gh1), "quantized words"
        for a, b, name in ((sg0, sg1, "sg"), (sh0, sh1, "sh")):
            assert b.numel() == S and torch.equal(a.view(torch.int64), b.contiguous().view(torch.int64)), name
        assert sg1.data_ptr() == scal.sg[s0:].data_ptr() and sh1.data_ptr() == scal.sh[s0:].data_ptr()
        assert (gh0 != 0).sum() > 1000 and (gh0[~w_d] == 0).all()


@pytest.mark.parametrize("qmax", ["segment", "triton", "scatter"])
def test_quantize_nm_edge_maxima(monkeypatch, qmax):
    from mangrove import kernels_gpu as KG, kernels_nm as KN
    monkeypatch.setattr(KG, "QMAX", qmax)
    rng = np.random.default_rng(4242)
    dev = torch.device("cuda")
    tiny, big = np.float32(1.401298464324817e-45), np.float32(np.finfo(np.float32).max)
    kinds = ["normal", "subnormal", "smallest", "maxfinite", "inf", "neg_inf", "nan", "zero"]
    S = 2 * len(kinds)
    lens = rng.integers(50, 400, S)
    rt_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    RT = int(rt_off[-1])
    slot = np.repeat(np.arange(S, dtype=np.int32), lens)
    work = rng.random(RT) < 0.8
    g = rng.standard_normal(RT).astype(np.float32)
    h = (np.abs(rng.standard_normal(RT)) + 0.01).astype(np.float32)
    for s, kind in enumerate(kinds * 2):
        a, b = int(rt_off[s]), int(rt_off[s + 1])
        work[a] = True
        if kind == "subnormal":
            g[a:b] = (rng.standard_normal(b - a) * 2.0 ** -133).astype(np.float32)
            h[a:b] = (rng.random(b - a) * 2.0 ** -130).astype(np.float32)
            g[a], h[a] = np.float32(2.0 ** -127), np.float32(2.0 ** -127)
        elif kind == "smallest":
            g[a:b] = rng.choice([-tiny, np.float32(0.0), tiny], b - a)
            h[a:b] = tiny
        elif kind == "maxfinite":
            g[a], h[a] = -big, big
        elif kind == "inf":
            g[a], h[a] = np.inf, np.inf
        elif kind == "neg_inf":
            g[a] = -np.inf
        elif kind == "nan":
            g[a], h[a] = np.nan, np.nan
        elif kind == "zero":
            g[a:b], h[a:b] = 0.0, 0.0
    bits = np.repeat(np.array([16, 28], dtype=np.int8), len(kinds))
    t = lambda x: torch.from_numpy(x).to(dev)
    g_d, h_d, sl_d, w_d, bits_d, ro_d = t(g), t(h), t(slot), t(work), t(bits), t(rt_off)
    for has_hess in (True, False):
        gh0, sg0, sh0 = KG.quantize(g_d, h_d, sl_d, w_d, bits_d, S, has_hess=has_hess, rt_off=ro_d)
        s0 = 3
        scal = KN.Scal(s0 + S + 1, dev)
        gh1, sg1, sh1 = KN.quantize_nm(KG, g_d, h_d, sl_d, w_d, bits_d, S, has_hess, ro_d, scal, s0)
        assert torch.equal(gh0, gh1), "quantized words"
        for x, y_, name in ((sg0, sg1, "sg"), (sh0, sh1, "sh")):
            assert torch.equal(x.view(torch.int64), y_.contiguous().view(torch.int64)), name
    gm, hm = (x.cpu().numpy() for x in KG._slot_maxima(g_d, h_d, sl_d, w_d, S, True, ro_d))
    tiny_n = np.finfo(np.float32).tiny
    for s, kind in enumerate(kinds * 2):
        for m, is_h in ((gm[s], False), (hm[s], True)):
            if kind == "nan":
                assert (np.isnan(m) if qmax != "triton" else np.isfinite(m) and m > 0), (kind, is_h, qmax, m)
            elif kind == "inf" or (kind == "neg_inf" and not is_h):
                assert np.isinf(m) and m > 0, (kind, is_h, m)
            elif kind == "subnormal":
                assert 0 < m < tiny_n, (kind, is_h, m)
            elif kind == "smallest":
                assert m == tiny, (kind, is_h, m)
            elif kind == "maxfinite":
                assert m == big, (kind, is_h, m)
            elif kind == "zero":
                assert m == 0, (kind, is_h, m)
            else:
                assert np.isfinite(m) and m >= tiny_n, (kind, is_h, m)


def _rowsum_case(rng, lens, work_frac):
    rt_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    RT = int(rt_off[-1])
    work = np.zeros(RT, dtype=bool)
    for s in range(len(lens)):
        work[rt_off[s]:rt_off[s + 1]] = rng.random(lens[s]) < work_frac[s]
    gq = rng.integers(-32767, 32768, RT).astype(np.int64)
    hq = rng.integers(0, 65536, RT).astype(np.int64)
    gh = np.where(work, gq * 4294967296 + hq, 0)
    return rt_off, work, gh


def test_rowsum_matches_node_sums_and_nm_init():
    from mangrove import kernels_gpu as KG, kernels_nm as KN
    rng = np.random.default_rng(77)
    dev = torch.device("cuda")
    lens = np.array([3000, 0, 200000, 1, 7, 50000, 2, 4096, 0])
    frac = np.array([0.6, 0.0, 0.9, 1.0, 0.5, 0.0, 1.0, 0.3, 0.0])
    S = len(lens)
    rt_off, work, gh = _rowsum_case(rng, lens, frac)
    RT = len(work)
    gh_dirty = gh.copy()
    gh_dirty[~work] = rng.integers(-(1 << 50), 1 << 50, RT)[~work]
    t = lambda a: torch.from_numpy(a).to(dev)
    P = KN.Plan(rt_off, np.arange(S))
    P.upload(lambda a: t(np.ascontiguousarray(a, dtype=np.int64)), 2048)
    assert P.n_chunks > S and P.E == 0
    scal = KN.Scal(S, dev)
    scal.buf.fill_(-1)
    for words in (gh, gh_dirty):
        scal.zero_sums()
        nm = torch.full((RT,), 7, dtype=torch.uint8, device=dev)
        KN.rowsum(t(words), t(work), nm, P, scal)
        ref = torch.full((RT,), 255, dtype=torch.uint8, device=dev)
        ref.masked_fill_(t(work), 0)
        assert torch.equal(nm, ref), "node-map init bytes"
        wl = torch.nonzero(t(work)).squeeze(1).to(torch.int32)
        wl_off = torch.searchsorted(wl, t(rt_off).to(torch.int32)).to(torch.int32)
        G, H = KG.node_sums(t(words), wl, wl_off[:-1], wl_off[1:])
        Gr, Hr, cnt, sg_r, sh_r = scal.readback()
        assert np.array_equal(Gr, G.cpu().numpy()) and np.array_equal(Hr, H.cpu().numpy()), "root sums"
        assert np.array_equal(cnt, (wl_off[1:] - wl_off[:-1]).cpu().numpy()), "work counts"
        assert cnt.dtype == np.int32 and Gr.dtype == np.int64
    assert cnt[2] > 65536 and Hr[2] > 2 ** 32 and (Gr < 0).any() and cnt[1] == 0 and cnt[5] == 0 and cnt[0] > 0
    assert (sg_r.view(np.int64) == -1).all() and (sh_r.view(np.int64) == -1).all()
    u = scal.buf.cpu().numpy()
    assert (u[:scal.w["G"][0]] == -1).all() and (u[scal.w["sg"][0]:] == -1).all()


def test_multi_row_groups_prologue(monkeypatch):
    orig = engine_module.Engine._round_groups

    def per_model(self, w_ub):
        orig(self, w_ub)
        groups = self._row_groups(1)
        return groups, max((b - a for _, _, a, b in groups), default=0)

    monkeypatch.setattr(engine_module.Engine, "_round_groups", per_model)
    rng = np.random.default_rng(61)
    Xb, y, _ = synth(2500, 7, nb=24, seed=12, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(2500, 4, rng, 7, depth=(3, 6), sub=(0.6, 1.0), col=(0.6, 1.0), rounds=5)
    for i, s in enumerate(specs):
        s.n_rounds = 2 + i % 4
    pair = _pair(co, "logistic", specs)
    _assert_same(pair)
    b = pair[1][0]
    assert b.S > 1
    assert b.nm_stats["multi_group_rounds"] == b.nm_stats["rounds"] > 0, b.nm_stats


def test_slot_ids_int32_matches_repeat_interleave():
    class _E:
        pass
    rng = np.random.default_rng(1107)
    dev = torch.device("cuda")
    for t in range(200):
        S = int(rng.integers(1, 80))
        lens = rng.integers(0, 64, S) * (rng.random(S) < 0.75)
        if t % 6 == 0:
            lens[:3] = 0
        if t % 4 == 0:
            lens[-3:] = 0
        e = _E()
        e.dev, e.RT = dev, int(lens.sum())
        e.h = {"rt_off": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)}
        ref = torch.repeat_interleave(torch.arange(S, device=dev), torch.from_numpy(lens).to(dev),
                                      output_size=e.RT).to(torch.int32)
        got = Engine._slot_ids_int32(e, S)
        assert got.dtype == torch.int32 and torch.equal(got, ref), (t, lens)


def test_split_sub_batches(monkeypatch):
    seen = []
    orig = engine_module.Engine._split_batches

    def small(self, split_bytes, n_cand_p, n_cand_w):
        nb, b = orig(self, split_bytes, n_cand_p, n_cand_w)
        seen.append((n_cand_p, n_cand_w))
        return {False: 3, True: 2}, b

    monkeypatch.setattr(engine_module.Engine, "_split_batches", small)
    rng = np.random.default_rng(47)
    Xb, y, _ = synth(3000, 6, nb=32, seed=14, task="binary")
    co = [Cohort("c", Xb, y)]
    rows = np.arange(3000)
    specs = [ModelSpec("c", rows[rows % 3 != i % 3], rows[rows % 3 == i % 3], eta=0.2, lam=1.0, mcw=1.0,
                       max_depth=int(rng.integers(4, 7)), n_rounds=3, seed=800 + i,
                       acc_mode="wide" if i % 2 == 0 else "auto") for i in range(6)]
    assert {s.resolve_mode() for s in specs} == {"wide", "packed"}
    pair = _pair(co, "logistic", specs, chunk_parents=3)
    _assert_same(pair)
    assert any(p > 3 for p, _ in seen) and any(w > 2 for _, w in seen), seen[:10]


def _pause_specs(n, rng, rounds=4):
    rows = np.arange(n)
    specs = []
    for i in range(7):
        k = i % 3
        tr, ev = rows[rows % 3 != k], rows[rows % 3 == k]
        if i == 4:
            tr = tr[rng.random(len(tr)) < 0.6]
        if i == 5:
            ev = ev[:0]
        if i == 6:
            ev = rng.permutation(ev)[:200]
        specs.append(ModelSpec("c", tr, ev, eta=0.2, lam=1.0, mcw=1.0, max_depth=int(rng.integers(3, 6)),
                               n_rounds=2 if i == 2 else rounds, subsample=0.9 if i % 2 else 1.0, seed=900 + i))
    return specs


def _ev_ref(e, mid):
    _, m = e.margins(mid)
    return m[e.reg[mid]._ev_pos].copy()


def _same_f32(a, b):
    return (a.dtype == np.float32 and a.shape == b.shape and a.flags.c_contiguous
            and np.array_equal(a.view(np.int32), b.view(np.int32)))


def _n_rt(e, mid):
    return len(e._layout(e.reg[mid].spec, None).rows)


def _pause_engine(co, loss, K, specs, mode):
    e = Engine(co, loss, n_classes=K, device="cuda", backend="gpu", nodemap=True, max_model_slots=6)
    ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs])
    assert e.reg[ids[6]].status == "pending"
    e.step(); e.step()
    assert e.reg[ids[2]].status == "done"
    pausing = [ids[1], ids[4], ids[5]]
    log = {}
    if mode != "none":
        before = {mid: _ev_ref(e, mid) for mid in pausing}
        n_ev0 = len(e.events)
        reserve = e.planner.reserve
        if mode == "fallback":
            e.planner.reserve = e.planner.budget
        e.pause(pausing + [ids[1], ids[6], ids[2]])
        e.planner.reserve = reserve
        ev = e.events[n_ev0:]
        assert [x["event"] for x in ev] == ["pause"] * 3 + ["compact"], ev
        assert [x["model"] for x in ev[:3]] == pausing
        if mode == "spill":
            e._spill_paused()
        dev_bytes = sum(4 * K * _n_rt(e, m) for m in pausing)
        for mid in pausing:
            s = e._paused[mid]
            if mode == "device":
                assert s["pred"] is None and s["pred_dev"].is_cuda and s["pred_dev"].dtype == torch.float32
                assert tuple(s["pred_dev"].shape) == (K, _n_rt(e, mid))
            else:
                assert s["pred_dev"] is None and isinstance(s["pred"], np.ndarray) and s["pred"].dtype == np.float32
                assert s["pred"].shape == (K, _n_rt(e, mid))
            em = e.eval_margin(mid)
            assert _same_f32(em, before[mid]) and em.base is None, mid
            assert em.shape == ((len(before[mid]),) if K == 1 else (len(before[mid]), K))
        assert e._paused_dev_bytes == (dev_bytes if mode == "device" else 0)
        assert e.memory_report()["paused_device_bytes"] == e._paused_dev_bytes
        log["acc_after_pause"] = e.acc.snapshot()
        e.retire([ids[1]])
        assert ids[1] not in e._paused and e.reg[ids[1]].status == "retired"
        assert e._paused_dev_bytes == (dev_bytes - 4 * K * _n_rt(e, ids[1]) if mode == "device" else 0)
        assert _same_f32(e.eval_margin(ids[1]), before[ids[1]])
        e.step()
        assert e._paused_dev_bytes == 0 and all(s["pred_dev"] is None for s in e._paused.values())
        e.resume([ids[4], ids[5]])
        assert ids[4] not in e._paused and ids[5] not in e._paused and e._paused_dev_bytes == 0
        log["acc_after_resume"] = e.acc.snapshot()
    else:
        e.retire([ids[1]])
        e.step()
    e.run()
    assert all(e.reg[m].status in ("done", "retired") for m in ids)
    return e, ids, log


@pytest.mark.parametrize("loss,K", [("logistic", 1), ("softmax", 3)])
def test_pause_resume_equivalence(loss, K):
    rng = np.random.default_rng(1105 + K)
    task = "binary" if K == 1 else "multiclass"
    Xb, y, _ = synth(1800, 6, nb=16, seed=51 + K, task=task, K=K)
    Xh = np.random.default_rng(3).integers(0, 16, (300, 6)).astype(np.uint8)
    co = [Cohort("c", Xb, y)]
    specs = _pause_specs(1800, rng)
    runs = {mode: _pause_engine(co, loss, K, specs, mode) for mode in ("none", "device", "spill", "fallback")}
    e0, ids0, _ = runs["none"]
    for mode in ("device", "spill", "fallback"):
        e, ids, _ = runs[mode]
        for a, b in zip(ids0, ids):
            assert obs(e0, a) == obs(e, b), (mode, b)
            assert np.array_equal(e0.predict(a, Xh).view(np.int32), e.predict(b, Xh).view(np.int32)), (mode, b)
        assert e.nm_stats["rounds"] > 0
    ed, es, ef = runs["device"][0], runs["spill"][0], runs["fallback"][0]
    assert ed.events == es.events == ef.events
    assert runs["device"][2] == runs["spill"][2] == runs["fallback"][2]
    assert ed.events != e0.events


import contextlib
import sys


@contextlib.contextmanager
def _count_calls(obj, name, callers):
    orig, own, n = getattr(obj, name), name in vars(obj), []

    def wrap(*a, **k):
        if sys._getframe(1).f_code.co_name in callers:
            n.append(1)
        return orig(*a, **k)

    setattr(obj, name, wrap)
    try:
        yield n
    finally:
        if own:
            setattr(obj, name, orig)
        else:
            delattr(obj, name)


def _fold_specs(n, folds, m, rounds, seed0, depth=3):
    rows = np.arange(n)
    return [ModelSpec("c", rows[rows % folds != i % folds], rows[rows % folds == i % folds], max_depth=depth,
                      n_rounds=rounds if np.isscalar(rounds) else rounds[i], eta=0.2, seed=seed0 + i) for i in range(m)]


def test_small_chunks_identical(monkeypatch):
    monkeypatch.setattr(engine_module, "PINNED_LIFECYCLE", False)
    Xb, y, _ = synth(800, 5, nb=12, seed=83, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _fold_specs(800, 4, 4, 3, 1040)
    out = {}
    for cap, n_gather, n_spill in ((64 << 20, 1, 1), (2000, 2, 0), (7000, 1, 2)):
        monkeypatch.setattr(Engine, "DEV_CHUNK_BYTES", cap)
        e = Engine(co, "logistic", device="cuda", backend="gpu", nodemap=True)
        ids = e.add_models(specs)
        e.step()
        with _count_calls(torch, "cat", ("_ev_gather",)) as ng:
            e.pause(ids)
        with _count_calls(torch, "cat", ("_spill_paused", "_spill_paused_unpinned")) as ns:
            e._spill_paused()
        assert (len(ng), len(ns)) == (n_gather, n_spill), (cap, len(ng), len(ns))
        em = [e.eval_margin(m) for m in ids]
        pred = [e._paused[m]["pred"] for m in ids]
        assert all(p.dtype == np.float32 and p.shape == (1, 800) and p.flags.c_contiguous for p in pred)
        e.resume(ids)
        e.run()
        out[cap] = (e, ids, em, pred)
    e0, ids0, em0, pred0 = out[64 << 20]
    for cap in (2000, 7000):
        e, ids, em, pred = out[cap]
        assert all(_same_f32(x, z) for x, z in zip(em, em0)), cap
        assert all(np.array_equal(x.view(np.int32), z.view(np.int32)) for x, z in zip(pred, pred0)), cap
        assert all(obs(e0, x) == obs(e, z) for x, z in zip(ids0, ids)), cap
        assert e.events == e0.events


@pytest.mark.parametrize("loss,K", [("logistic", 1), ("softmax", 3)])
def test_pinned_lifecycle_identical(loss, K, monkeypatch):
    rng = np.random.default_rng(2207 + K)
    task = "binary" if K == 1 else "multiclass"
    Xb, y, _ = synth(1800, 6, nb=16, seed=61 + K, task=task, K=K)
    Xh = np.random.default_rng(5).integers(0, 16, (300, 6)).astype(np.uint8)
    co = [Cohort("c", Xb, y)]
    specs = _pause_specs(1800, rng)
    for mode in ("device", "spill", "fallback"):
        runs = {}
        for pinned in (False, True):
            monkeypatch.setattr(engine_module, "PINNED_LIFECYCLE", pinned)
            runs[pinned] = _pause_engine(co, loss, K, specs, mode)
        (a, ia, la), (b, ib, lb) = runs[False], runs[True]
        for x, z in zip(ia, ib):
            assert obs(a, x) == obs(b, z), (mode, z)
            assert np.array_equal(a.predict(x, Xh).view(np.int32), b.predict(z, Xh).view(np.int32)), (mode, z)
        assert a.events == b.events and la == lb, mode
        assert a.pin_stats["pin_models"] == 0 and a.pin_stats["resume_pinned"] == 0
        assert b.pin_stats["pin_models"] > 0 and b.pin_stats["resume_pinned"] > 0, (mode, b.pin_stats)


def _grad_margins(rng, n):
    x = (rng.standard_normal(n) * rng.choice([0.5, 3.0, 20.0], n)).astype(np.float32)
    m = rng.random(n)
    specials = np.array([80, -80, 1e4, -1e4, 3e38, -3e38, 0.0, -0.0, np.inf, -np.inf, 1e-40, -1e-40, 1e-30, 88.7, -88.7,
                         103.9, -103.9, 709.8, -709.8, 745.2, -745.2, 1.0, -1.0, 16.5, -16.5], dtype=np.float32)
    sel = m < 0.2
    x[sel] = rng.choice(specials, int(sel.sum()))
    nan_bits = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FC01234, 0xFFFFFFFF, 0x7FBFFFFF], dtype=np.uint32)
    sel = (m >= 0.2) & (m < 0.22)
    x.view(np.uint32)[sel] = rng.choice(nan_bits, int(sel.sum()))
    return x


def _mrow_like_engine(lens, K, a, dev):
    import types
    seg = np.repeat(np.asarray(lens, dtype=np.int64), K)
    rt_off = np.concatenate([[0], np.cumsum(seg)]).astype(np.int64) + a
    ns = types.SimpleNamespace(dev=dev, K=K, S=len(seg), RT=int(rt_off[-1]), loss="softmax",
                               rt_off=torch.from_numpy(rt_off).to(dev))
    return engine_module.Engine._mrow(ns, 0, len(seg), a, int(rt_off[-1]))


def _assert_same_bits(name, got, ref, pred):
    bad = torch.nonzero(got.view(torch.int32) != ref.view(torch.int32)).flatten()
    assert len(bad) == 0, (name, len(bad), bad[:5].tolist(), pred[bad[:5]].tolist(), got[bad[:5]].tolist(),
                           ref[bad[:5]].tolist())


@pytest.mark.parametrize("loss,K", [("logistic", 1), ("softmax", 1), ("softmax", 3), ("softmax", 7), ("softmax", 10)])
def test_grad_hess_nm_matches_torch(loss, K):
    from mangrove import kernels_nm as KN, kernels_ref as KR
    rng = np.random.default_rng(7000 + 10 * K + (loss == "softmax"))
    dev = torch.device("cuda")
    lens = [1, 2, 37, 1000, 5, 70001, 3, 64, 2049]
    a = 1234
    rows = int(sum(lens))
    n = rows * K
    assert n % KN.GRAD_THREADS, "n must not be a multiple of the block size"
    big = torch.from_numpy(_grad_margins(rng, a + n + 77)).to(dev)
    pred = big[a:a + n]
    assert pred.is_contiguous() and pred.storage_offset() == a
    if loss == "logistic":
        mrow_base = mrow_len = None
        yv = rng.choice(np.array([0.0, 1.0], dtype=np.float32), n)
        odd = rng.random(n) < 0.01
        yv[odd] = rng.choice(np.array([0.5, -0.0, 2.0, np.inf], dtype=np.float32), int(odd.sum()))
        odd = rng.random(n) < 0.005
        yv.view(np.uint32)[odd] = rng.choice(np.array([0x7FC00000, 0xFFC00000], dtype=np.uint32), int(odd.sum()))
        y = torch.from_numpy(yv).to(dev)
    else:
        mrow_base, mrow_len = _mrow_like_engine(lens, K, a, dev)
        eb, el, off = [], [], 0
        for L in lens:
            eb.append(np.arange(off, off + L))
            el.append(np.full(L, L if K > 1 else 0))
            off += K * L
        assert np.array_equal(mrow_base.cpu().numpy(), np.concatenate(eb))
        assert np.array_equal(mrow_len.cpu().numpy(), np.concatenate(el))
        Nm = rows
        idx = mrow_base[None, :] + torch.arange(K, device=dev, dtype=torch.int64)[:, None] * mrow_len[None, :]
        if K > 1:
            parts = torch.chunk(torch.from_numpy(rng.permutation(Nm)[:2000]).to(dev), 8)
            pred[idx[:, parts[0]]] = pred[idx[0, parts[0]]][None, :]
            pred[idx[1, parts[1]]] = pred[idx[0, parts[1]]]
            pred[idx[K - 1, parts[1]]] = pred[idx[0, parts[1]]]
            zs = torch.tensor([0.0, -0.0], device=dev)
            pred[idx[:, parts[2]]] = zs[torch.randint(0, 2, (K, len(parts[2])), device=dev)]
            pred[idx[:, parts[3]]] = float("-inf")
            pred[idx[K // 2, parts[4]]] = float("inf")
            pred[idx[:, parts[5]]] = float("inf")
            nans = torch.tensor([0x7FC00000, -4194304, 0x7FC01234, -1], dtype=torch.int32, device=dev).view(torch.float32)
            pred[idx[K - 1, parts[6]]] = nans[0]
            pred[idx[0, parts[7]]] = nans[2]
            pred[idx[1, parts[7]]] = nans[3]
            if K > 2:
                pred[idx[2, parts[7]]] = nans[1]
        yk = rng.integers(0, K, Nm).astype(np.int64)
        yk[:K] = np.arange(K)
        oor = rng.random(Nm) < 0.01
        yk[oor] = rng.choice(np.array([-1, K, 2 * K + 3], dtype=np.int64), int(oor.sum()))
        yk = torch.from_numpy(yk).to(dev)
        y = torch.empty(n, dtype=torch.int64, device=dev)
        y[idx.reshape(-1)] = yk.repeat(K)
    gr, hr = KR.grad_hess(loss, K, pred, y, mrow_base, mrow_len)
    out = KN.grad_hess_nm(loss, K, pred, y, mrow_base, mrow_len)
    assert out is not None
    g, h = out
    assert g.dtype == h.dtype == torch.float32 and g.shape == h.shape == pred.shape and g.device == pred.device
    _assert_same_bits("g", g, gr, pred)
    _assert_same_bits("h", h, hr, pred)
    pv = pred.cpu().numpy()
    assert np.isnan(pv).any() and np.isposinf(pv).any() and np.isneginf(pv).any()
    assert (np.abs(pv) >= 3e38).any() and ((np.abs(pv) > 0) & (np.abs(pv) < np.finfo(np.float32).tiny)).any()
    assert ((pv == 0) & np.signbit(pv)).any()
    assert bool(torch.isnan(gr).any()) and bool((hr == np.float32(1e-16)).any())
    assert (K == 1 and loss == "softmax") or bool((hr > np.float32(1e-16)).any())
    assert KN.grad_hess_nm("squared", K, pred, y.to(torch.float32), mrow_base, mrow_len) is None


def test_grad_hess_nm_unsupported_returns_none():
    from mangrove import kernels_nm as KN
    dev = torch.device("cuda")
    n = 1000
    pred = torch.randn(n, device=dev)
    y32 = torch.zeros(n, device=dev)
    y64 = torch.zeros(n, dtype=torch.int64, device=dev)
    base, ln = torch.arange(n, device=dev), torch.zeros(n, dtype=torch.int64, device=dev)
    assert KN.grad_hess_nm("squared", 1, pred, y32, None, None) is None
    assert KN.grad_hess_nm("logistic", 1, pred.cpu(), y32.cpu(), None, None) is None
    assert KN.grad_hess_nm("logistic", 1, pred[::2], y32[::2], None, None) is None
    assert KN.grad_hess_nm("logistic", 1, pred, y64, None, None) is None
    assert KN.grad_hess_nm("logistic", 1, pred.double(), y32, None, None) is None
    assert KN.grad_hess_nm("logistic", 1, pred, y32[:n - 1], None, None) is None
    assert KN.grad_hess_nm("softmax", 1, pred, y32, base, ln) is None
    assert KN.grad_hess_nm("softmax", 33, pred, y64, base, ln) is None
    assert KN.grad_hess_nm("softmax", 2, pred, y64, base, ln) is None
    assert KN.grad_hess_nm("softmax", 1, pred, y64, base.to(torch.int32), ln) is None
    assert KN.grad_hess_nm("softmax", 1, pred, y64, base, ln[:n - 1]) is None
    assert KN.grad_hess_nm("softmax", 1, pred, y64, None, None) is None
    assert KN.grad_hess_nm("logistic", 1, pred, y32, None, None) is not None
    assert KN.grad_hess_nm("softmax", 1, pred, y64, base, ln) is not None
    g, h = KN.grad_hess_nm("logistic", 1, pred[:0], y32[:0], None, None)
    assert g.numel() == 0 and h.numel() == 0 and g.dtype == h.dtype == torch.float32


def test_grad_hess_nm_logistic_exhaustive():
    from mangrove import kernels_nm as KN, kernels_ref as KR
    dev = torch.device("cuda")
    CH = 1 << 26
    y = torch.zeros(CH, dtype=torch.float32, device=dev)
    bad_g = bad_h = 0
    for c in range(1 << 6):
        v = torch.arange(c * CH, (c + 1) * CH, dtype=torch.int64, device=dev)
        pred = torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32).view(torch.float32)
        del v
        gr, hr = KR.grad_hess("logistic", 1, pred, y, None, None)
        g, h = KN.grad_hess_nm("logistic", 1, pred, y, None, None)
        bad_g += int((g.view(torch.int32) != gr.view(torch.int32)).sum())
        bad_h += int((h.view(torch.int32) != hr.view(torch.int32)).sum())
        del gr, hr, g, h, pred
    assert bad_g == 0 and bad_h == 0, (bad_g, bad_h)


@pytest.mark.parametrize("loss,K,task", [("logistic", 1, "binary"), ("softmax", 3, "multiclass"),
                                         ("softmax", 7, "multiclass"), ("squared", 1, "regression")])
def test_engine_uses_fused_gradients(loss, K, task, monkeypatch):
    from mangrove import kernels_nm
    Xb, y, _ = synth(1500, 6, nb=14, seed=113 + K, task=task, K=K)
    co = [Cohort("c", Xb, y)]
    specs = _fold_specs(1500, 5, 7, 4, 1200, depth=4)
    calls = []
    orig = kernels_nm.grad_hess_nm

    def spy(*a, **k):
        r = orig(*a, **k)
        calls.append(r is not None)
        return r
    res = {}
    for arm in ("fused", "torch", "p7off"):
        monkeypatch.setattr(engine_module, "FUSED_GRADIENTS", arm != "torch")
        monkeypatch.setattr(engine_module, "PINNED_LIFECYCLE", arm != "p7off")
        monkeypatch.setattr(kernels_nm, "grad_hess_nm", spy)
        calls.clear()
        e = Engine(co, loss, n_classes=K, device="cuda", backend="gpu", nodemap=True)
        ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs])
        e.run()
        assert e.nm_stats["rounds"] > 0
        if arm == "fused":
            assert len(calls) >= e.nm_stats["rounds"] and all(c == (loss != "squared") for c in calls), (len(calls), calls[:3])
        else:
            assert calls == [], (arm, len(calls))
        res[arm] = [obs(e, m) for m in ids]
    assert res["fused"] == res["torch"] == res["p7off"]


_I64 = np.iinfo(np.int64)
_EDGE_SEEDS = np.array([0, 1, -1, 7, _I64.max, _I64.min, _I64.min + 1, 2 ** 32, 2 ** 32 - 1, -2 ** 31, 2 ** 31 - 1,
                        2 ** 40 + 3, 1000, 1592521082], dtype=np.int64)
_EDGE_LR = np.array([0, 1, 2, 99, 2 ** 31 - 1, 2 ** 33, -5], dtype=np.int64)
_EDGE_SUB = np.array([0.0, 1e-9, 0.3, 0.5, 0.7, 0.9999999, 1.0, 1.0000001, 1.5, 0.8, 1 - 2.0 ** -33], dtype=np.float64)


def _rowmask_torch(is_train, active, slot_of_rt, seed, lr, sub, crow, s0, s1, a, b, samp):
    from mangrove import kernels_ref as KR
    S = active.numel()
    whole = s0 == 0 and s1 == S
    sl = slot_of_rt if whole else slot_of_rt[a:b] - s0
    pv = (lambda t: t) if whole else (lambda t: t[s0:s1])
    rv = (lambda t: t) if whole else (lambda t: t[a:b])
    w = rv(is_train) & pv(active).index_select(0, sl)
    if samp:
        sl64 = sl.to(torch.int64)
        tag = torch.full((), KR.ROW_STREAM, device=is_train.device, dtype=torch.int64)
        u = KR.hash_u01(tag, pv(seed)[sl64], pv(lr)[sl64], rv(crow).to(torch.int64))
        w = w & (u < pv(sub)[sl64])
    return w


def _rowmask_layout(rng, S, lens):
    dev = torch.device("cuda")
    rt_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    RT = int(rt_off[-1])
    seed = rng.choice(_EDGE_SEEDS, S)
    lr = rng.choice(_EDGE_LR, S)
    sub = rng.choice(_EDGE_SUB, S)
    act = rng.random(S) < 0.8
    act[0] = True
    slot = np.repeat(np.arange(S, dtype=np.int32), lens)
    is_train = rng.random(RT) < 0.75
    crow = rng.integers(-2 ** 31, 2 ** 31, RT).astype(np.int32)
    ext = rng.random(RT) < 0.02
    crow[ext] = rng.choice(np.array([-2 ** 31, 2 ** 31 - 1, 0, -1, 1], dtype=np.int32), int(ext.sum()))
    t = lambda x, dt: torch.from_numpy(np.ascontiguousarray(x)).to(dt).to(dev)
    return dict(S=S, RT=RT, rt_off=rt_off, seed_h=seed, lr_h=lr, sub_h=sub, act_h=act,
                seed=t(seed, torch.int64), lr=t(lr, torch.int64), sub=t(sub, torch.float64), act=t(act, torch.bool),
                slot=t(slot, torch.int32), is_train=t(is_train, torch.bool), crow=t(crow, torch.int32))


def test_rowmask_matches_torch_sampling():
    from mangrove import kernels_nm as KN, kernels_ref as KR
    import json
    from pathlib import Path
    rng = np.random.default_rng(8001)
    dev = torch.device("cuda")
    S = 23
    lens = rng.integers(1, 3000, S)
    lens[3] = 1
    lens[7] = 70001
    L = _rowmask_layout(rng, S, lens)
    tab_h = KN.rowmask_tab(L["seed_h"], L["lr_h"], L["sub_h"], L["act_h"])
    assert tab_h.dtype == np.int64 and tab_h.shape == (4 * S,)
    assert np.array_equal(tab_h[2 * S:3 * S].view(np.float64), L["sub_h"])
    assert np.array_equal(tab_h[3 * S:] & 1, L["act_h"].astype(np.int64))
    assert np.array_equal((tab_h[3 * S:] >> 1) & 1, (L["sub_h"] < 1.0).astype(np.int64))
    tab = torch.from_numpy(tab_h).to(dev)
    ro = L["rt_off"]
    groups = [(0, S, 0, L["RT"])] + [(s, s + 1, int(ro[s]), int(ro[s + 1])) for s in range(S)]
    groups += [(2, 9, int(ro[2]), int(ro[9])), (10, 23, int(ro[10]), int(ro[23])), (0, 12, 0, int(ro[12]))]
    n_samp = n_plain = 0
    for s0, s1, a, b in groups:
        samp = bool(((L["sub_h"] < 1.0) & L["act_h"])[s0:s1].any())
        ref = _rowmask_torch(L["is_train"], L["act"], L["slot"], L["seed"], L["lr"], L["sub"], L["crow"], s0, s1, a, b, samp)
        w = torch.empty(b - a, dtype=torch.bool, device=dev)
        u = torch.empty(b - a, dtype=torch.float64, device=dev)
        KN.rowmask(L["is_train"], L["crow"], L["slot"], a, b - a, tab, S, w, u_out=u)
        assert torch.equal(w, ref), (s0, s1, a, b, samp)
        w2 = torch.empty(b - a, dtype=torch.bool, device=dev)
        KN.rowmask(L["is_train"], L["crow"], L["slot"], a, b - a, tab, S, w2)
        assert torch.equal(w2, ref)
        tag = torch.full((), KR.ROW_STREAM, device=dev, dtype=torch.int64)
        sl64 = L["slot"][a:b].to(torch.int64)
        uref = KR.hash_u01(tag, L["seed"][sl64], L["lr"][sl64], L["crow"][a:b].to(torch.int64))
        assert torch.equal(u.view(torch.int64), uref.view(torch.int64)), (s0, s1)
        n_samp += samp
        n_plain += not samp
    assert n_samp > 0 and n_plain > 0
    w = torch.empty(L["RT"], dtype=torch.bool, device=dev)
    KN.rowmask(L["is_train"], L["crow"], L["slot"], 0, L["RT"], tab, S, w)
    base = L["is_train"] & L["act"].index_select(0, L["slot"])
    for s in range(S):
        ws, bs = w[ro[s]:ro[s + 1]], base[ro[s]:ro[s + 1]]
        if not L["act_h"][s]:
            assert not bool(ws.any())
        elif L["sub_h"][s] >= 1.0:
            assert torch.equal(ws, bs)
        elif L["sub_h"][s] == 0.0:
            assert not bool(ws.any())
        else:
            assert bool((ws <= bs).all()) and int(ws.sum()) <= int(bs.sum())
    kat = [k for k in json.load(open(Path(__file__).resolve().parent / "hash_kat.json")) if k["keys"][0] == KR.ROW_STREAM]
    assert len(kat) >= 8
    n = len(kat)
    seed_h = np.array([k["keys"][1] for k in kat], dtype=np.int64)
    lr_h = np.array([k["keys"][2] for k in kat], dtype=np.int64)
    crow_h = np.array([k["keys"][3] for k in kat], dtype=np.int32)
    assert np.array_equal(crow_h.astype(np.int64), np.array([k["keys"][3] for k in kat]))
    tab = torch.from_numpy(KN.rowmask_tab(seed_h, lr_h, np.full(n, 0.5), np.ones(n, dtype=bool))).to(dev)
    u = torch.empty(n, dtype=torch.float64, device=dev)
    w = torch.empty(n, dtype=torch.bool, device=dev)
    KN.rowmask(torch.ones(n, dtype=torch.bool, device=dev), torch.from_numpy(crow_h).to(dev),
               torch.arange(n, dtype=torch.int32, device=dev), 0, n, tab, n, w, u_out=u)
    assert u.view(torch.int64).cpu().tolist() == [k["u_bits"] for k in kat]
    assert w.cpu().tolist() == [k["u"] < 0.5 for k in kat]
    KN.rowmask(L["is_train"], L["crow"], L["slot"], 5, 0, tab, n, torch.empty(0, dtype=torch.bool, device=dev))


def _colsample_ns(S, d, seed_h, lr_h):
    import types
    from mangrove import kernels_gpu as KG
    dev = torch.device("cuda")
    return types.SimpleNamespace(S=S, d=d, dev=dev, KR=KG, seed=torch.from_numpy(np.asarray(seed_h, dtype=np.int64)).to(dev),
                                 local_round=torch.from_numpy(np.asarray(lr_h, dtype=np.int64)).to(dev))


def test_colsample_host_matches_torch_sampling():
    from mangrove import kernels_nm as KN
    import json
    from pathlib import Path
    rng = np.random.default_rng(8002)
    dev = torch.device("cuda")
    for item in json.load(open(Path(__file__).resolve().parent / "hash_kat.json")):
        h = KN.hash_u32_np(*item["keys"])
        assert np.float64(float(h) / 4294967296.0).view(np.int64) == item["u_bits"], item
    n_cases = 0
    for trial in range(120):
        S, d = int(rng.integers(1, 70)), int(rng.integers(1, 90))
        seed_h = rng.choice(_EDGE_SEEDS, S)
        lr_h = rng.choice(_EDGE_LR, S)
        frac_h = rng.choice(np.array([0.0, 1e-9, 0.1, 0.5, 0.7, 0.5000001, 0.9999999, 1.0, 1.5, 1 / 3, 2 / 3, 0.8,
                                      np.inf, -np.inf, np.nan, 1e30, 2.0 ** 63 / 20]), S)
        lk = int(rng.choice([0, 1, 2, 8]))
        ns = _colsample_ns(S, d, seed_h, lr_h)
        if lk == 0:
            base_h = np.ones((S, d), dtype=bool)
            base_arg = None if trial % 2 else base_h
        else:
            base_h = rng.random((S, d)) < rng.choice([0.0, 0.3, 0.6, 1.0])
            base_arg = base_h
        ref = engine_module.Engine._colsample_mask(ns, lk, torch.from_numpy(base_h).to(dev),
                                                   torch.from_numpy(frac_h).to(dev))
        got = KN.colsample_host(seed_h, lr_h, lk, base_arg, frac_h, d=d)
        assert got.dtype == np.uint8 and got.shape == (S, d)
        assert np.array_equal(got.astype(bool), ref.cpu().numpy()), (trial, S, d, lk)
        n_cases += 1
        if lk == 0:
            with np.errstate(invalid="ignore", over="ignore"):
                k = np.maximum(np.ceil(np.minimum(frac_h * d, float(d))).astype(np.int64), 1)
            assert np.array_equal(got.sum(1), np.minimum(k, d))
            assert np.array_equal(got.sum(1)[np.isposinf(frac_h) | (frac_h >= 1e30)], np.full(int((np.isposinf(frac_h) | (frac_h >= 1e30)).sum()), d))
            assert np.array_equal(got.sum(1)[np.isnan(frac_h) | np.isneginf(frac_h)], np.ones(int((np.isnan(frac_h) | np.isneginf(frac_h)).sum()), dtype=np.int64))
    assert n_cases == 120


def _sampling_specs(n, rng, M, folds=4, rounds=5):
    specs = _cv_specs(n, folds, rng, M, depth=(3, 7), sub=(0.5, 1.0), col=(0.5, 1.0), rounds=rounds)
    for i, s in enumerate(specs):
        if i % 4 == 0:
            s.subsample = 1.0
        if i % 5 == 1:
            s.colsample_bytree = 1.0
        if i % 3 == 2:
            s.colsample_bylevel = float(rng.uniform(0.4, 0.9))
        if i == 3:
            s.subsample, s.colsample_bytree = 0.5, 0.5
    return specs


def _sampling_arms(co, loss, K, specs, monkeypatch, lifecycle=False, debug_checks=True, **kw):
    out = {}
    for arm in ("worklist", "torch_sampling", "fused_sampling"):
        monkeypatch.setattr(engine_module, "FUSED_SAMPLING", arm == "fused_sampling")
        monkeypatch.setattr(engine_module, "DEBUG_CHECKS", debug_checks and arm == "fused_sampling")
        e = Engine(co, loss, n_classes=K, device="cuda", backend="gpu", nodemap=arm != "worklist", **kw)
        first = specs[:7] if lifecycle else specs
        ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in first])
        if lifecycle:
            e.step(); e.step()
            e.pause([ids[1]])
            e.retire([ids[2]])
            e.step()
            e.resume([ids[1]])
            e.compact()
            ids += e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs[7:]])
        e.run()
        out[arm] = ([obs(e, m) for m in ids], e)
    monkeypatch.setattr(engine_module, "DEBUG_CHECKS", False)
    return out


def _assert_sampling_arms(arms, rows=True, cols=True):
    a, b, c = arms["worklist"], arms["torch_sampling"], arms["fused_sampling"]
    assert a[0] == b[0] == c[0]
    assert a[1].nm_stats["rounds"] == 0 and b[1].nm_stats["rounds"] > 0 and c[1].nm_stats["rounds"] > 0
    assert b[1].nm_stats["worklist_rounds"] == c[1].nm_stats["worklist_rounds"] == 0
    assert b[1].nm_stats["fused_rowmask"] == b[1].nm_stats["fused_colmask"] == 0, b[1].nm_stats
    assert (c[1].nm_stats["fused_rowmask"] > 0) == rows and (c[1].nm_stats["fused_colmask"] > 0) == cols, c[1].nm_stats
    assert a[1].events == b[1].events == c[1].events


@pytest.mark.parametrize("loss,K", [("logistic", 1), ("squared", 1), ("softmax", 3)])
def test_fused_sampling_identical(loss, K, monkeypatch):
    rng = np.random.default_rng(8100 + K)
    task = {"logistic": "binary", "squared": "regression", "softmax": "multiclass"}[loss]
    Xb, y, _ = synth(3000, 9, nb=40, seed=81, task=task, K=K)
    co = [Cohort("c", Xb, y)]
    _assert_sampling_arms(_sampling_arms(co, loss, K, _sampling_specs(3000, rng, 12), monkeypatch))


def test_fused_sampling_lifecycle_chunks_slots_budget(monkeypatch):
    rng = np.random.default_rng(8200)
    Xb, y, _ = synth(2500, 7, nb=24, seed=82, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _sampling_specs(2500, rng, 11, folds=5, rounds=6)
    _assert_sampling_arms(_sampling_arms(co, "logistic", 1, specs, monkeypatch, lifecycle=True, max_model_slots=4))
    _assert_sampling_arms(_sampling_arms(co, "logistic", 1, specs, monkeypatch, chunk_parents=1))
    orig = engine_module.Engine._round_groups

    def per_model(self, w_ub):
        orig(self, w_ub)
        groups = self._row_groups(1)
        return groups, max((b - a for _, _, a, b in groups), default=0)

    monkeypatch.setattr(engine_module.Engine, "_round_groups", per_model)
    arms = _sampling_arms(co, "logistic", 1, specs, monkeypatch)
    _assert_sampling_arms(arms)
    e = arms["fused_sampling"][1]
    assert e.nm_stats["multi_group_rounds"] == e.nm_stats["rounds"] > 0, e.nm_stats


def test_fused_sampling_rows_only_cols_only(monkeypatch):
    rng = np.random.default_rng(8300)
    Xb, y, _ = synth(2400, 8, nb=24, seed=83, task="binary")
    co = [Cohort("c", Xb, y)]
    rows = _cv_specs(2400, 4, rng, 8, depth=(3, 6), sub=(0.5, 0.9), rounds=4)
    _assert_sampling_arms(_sampling_arms(co, "logistic", 1, rows, monkeypatch), rows=True, cols=False)
    cols = _cv_specs(2400, 4, rng, 8, depth=(3, 6), col=(0.4, 0.9), rounds=4)
    _assert_sampling_arms(_sampling_arms(co, "logistic", 1, cols, monkeypatch), rows=False, cols=True)
    lvl = _cv_specs(2400, 4, rng, 8, depth=(3, 6), rounds=4)
    for i, s in enumerate(lvl):
        s.colsample_bylevel = (0.4, 0.6, 0.8, 1.0)[i % 4]
    _assert_sampling_arms(_sampling_arms(co, "logistic", 1, lvl, monkeypatch), rows=False, cols=True)


def test_retry_natural_low_budget(monkeypatch):
    from mangrove.memory import LevelInfeasible
    Xb, y, _ = synth(800, 6, nb=16, seed=4242, task="regression")
    co = [Cohort("g", Xb, y)]
    rng = np.random.default_rng(99)
    specs = [ModelSpec("g", rng.permutation(800)[:64], [], n_rounds=3, max_depth=7, mcw=0.0, lam=0.01, seed=i + 11,
                       acc_mode="wide", colsample_bytree=0.5) for i in range(4)]

    def run(nodemap, budget, fsamp):
        monkeypatch.setattr(engine_module, "FUSED_SAMPLING", fsamp)
        e = Engine(co, "squared", device="cuda", backend="gpu", nodemap=nodemap, budget_bytes=budget, collect_metrics=False)
        ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs])
        rec = dict(core_open=0, retries=0)
        orig = e._plan_core

        def plan(*a):
            try:
                return orig(*a)
            except LevelInfeasible:
                r = getattr(e, "_nm_ring", None)
                if r is not None and r["open"]:
                    rec["core_open"] += 1
                raise
        e._plan_core = plan
        o_degrade = e._degrade

        def degrade(*a, **k):
            out = o_degrade(*a, **k)
            r = getattr(e, "_nm_ring", None)
            assert r is None or not r["open"], "the failed round left the ring open"
            if any(e.reg[m].status == "active" for m in ids):
                rec["retries"] += 1
            return out
        e._degrade = degrade
        e.run()
        return [obs(e, m) for m in ids], rec, e

    ref = run(False, 2 ** 40, False)[0]
    st = dict(budgets=0, mem=0, retry_budgets=0, evict_budgets=0)
    b = 45e3
    while b < 140e3:
        budget = int(b)
        b *= 1.03
        st["budgets"] += 1
        try:
            on, rec, e = run(True, budget, True)
            off, _, _ = run(True, budget, False)
        except MemoryError:
            st["mem"] += 1
            continue
        assert on == ref and off == ref, (budget, rec, e.n_evictions)
        st["retry_budgets"] += rec["core_open"] > 0 and rec["retries"] > 0
        st["evict_budgets"] += e.n_evictions > 0
    assert st["retry_budgets"] >= 1 and st["evict_budgets"] >= 1, st


def test_fused_sampling_off_restores_torch_sampling(monkeypatch):
    from mangrove import kernels_gpu as KG
    calls = dict(rowmask=0, colmask=0, hash=0)
    o_rm, o_cm, o_h = Engine._nm_rowmask, Engine._nm_colsample, KG.hash_u01

    def rm(self, *a):
        calls["rowmask"] += 1
        return o_rm(self, *a)

    def cm(self, *a):
        calls["colmask"] += 1
        return o_cm(self, *a)

    def h(*k):
        calls["hash"] += 1
        return o_h(*k)
    monkeypatch.setattr(Engine, "_nm_rowmask", rm)
    monkeypatch.setattr(Engine, "_nm_colsample", cm)
    monkeypatch.setattr(KG, "hash_u01", h)
    rng = np.random.default_rng(8500)
    Xb, y, _ = synth(1500, 6, nb=16, seed=85, task="binary")
    co = [Cohort("c", Xb, y)]
    specs = _cv_specs(1500, 3, rng, 6, depth=(3, 5), sub=(0.5, 0.9), col=(0.5, 0.9), rounds=4)
    res = {}
    for fsamp in (True, False):
        monkeypatch.setattr(engine_module, "FUSED_SAMPLING", fsamp)
        for k in calls:
            calls[k] = 0
        e = Engine(co, "logistic", device="cuda", backend="gpu", nodemap=True)
        ids = e.add_models([ModelSpec(**{f: getattr(s, f) for f in s.__dataclass_fields__}) for s in specs])
        e.run()
        assert e.nm_stats["rounds"] == 4 and e.nm_stats["worklist_rounds"] == 0
        if fsamp:
            assert calls["rowmask"] == 4 and calls["colmask"] == 4 and calls["hash"] == 0, calls
        else:
            assert calls["rowmask"] == calls["colmask"] == 0 and calls["hash"] == 8, calls
        res[fsamp] = [obs(e, m) for m in ids]
    assert res[True] == res[False]
