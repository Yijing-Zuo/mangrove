import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULTS = dict(n=50_000, d=32, informative=0.5, task="binary", n_classes=2, noise=0.05, n_teachers=8,
                teacher_depth=6, skew=0.0, leaf_prob=0.0, alpha=0.0, frac_lowcard=0.0, n_categorical=0, cat_max=40,
                corr=0.0, missing_rate=0.0, pos_rate=0.5, n_groups=0, group_skew=1.0, group_effect=0.5, seed=0)
CHUNK = 65_536
DESIGN_N = 20_000

INSTANCES = {
    "synth-balanced-50k": dict(skew=0.0),
    "synth-skewed-50k": dict(skew=0.9),
    "synth-mixed-50k": dict(skew=0.6, leaf_prob=0.15, alpha=0.3, frac_lowcard=0.25, n_categorical=4,
                            missing_rate=0.05, corr=0.3),
    "synth-reg-50k": dict(task="regression", skew=0.5, noise=0.3),
    "synth-mc5-20k": dict(n=20_000, task="multiclass", n_classes=5, skew=0.3, noise=0.05),
    "synth-grouped-50k": dict(skew=0.5, n_groups=32, group_skew=1.0, group_effect=0.5),
}


def _feature_plan(p, rng):
    d = p["d"]
    n_inf = max(1, int(round(p["informative"] * d)))
    kinds = np.array(["num"] * d, dtype=object)
    perm = rng.permutation(d)
    n_cat = min(p["n_categorical"], d)
    n_low = min(int(round(p["frac_lowcard"] * d)), d - n_cat)
    kinds[perm[:n_cat]] = "cat"
    kinds[perm[n_cat:n_cat + n_low]] = "low"
    levels = np.where(kinds == "low", rng.integers(2, 33, d), 0)
    card = np.where(kinds == "cat", rng.integers(3, max(4, p["cat_max"] + 1), d), 0)
    informative = np.sort(rng.permutation(d)[:n_inf])
    return dict(kinds=kinds, levels=levels, card=card, informative=informative)


def _gen_rows(p, plan, rng, n):
    d = p["d"]
    common = rng.standard_normal(n)
    Z = rng.standard_normal((n, d))
    c = p["corr"]
    Z = np.sqrt(1 - c) * Z + np.sqrt(c) * common[:, None]
    X = Z.copy()
    for j in range(d):
        k = plan["kinds"][j]
        if k == "low":
            L = plan["levels"][j]
            u = _norm_cdf(Z[:, j])
            X[:, j] = np.minimum((u * L).astype(np.int64), L - 1)
        elif k == "cat":
            K = plan["card"][j]
            w = 1.0 / np.arange(1, K + 1) ** 1.2
            cw = np.cumsum(w / w.sum())
            X[:, j] = np.minimum(np.searchsorted(cw, _norm_cdf(Z[:, j]), side="right"), K - 1)
    return X


def _norm_cdf(z):
    from scipy.special import ndtr
    return ndtr(z)


def _build_teacher(Xd, feats, rng, depth, skew, leaf_prob):
    feat, thr, left, right, value = [], [], [], [], []

    def new_node():
        feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); value.append(rng.standard_normal())
        return len(feat) - 1

    root = new_node()
    frontier = [(root, np.arange(len(Xd)), 0)]
    while frontier:
        nxt = []
        for nid, rows, lv in frontier:
            if lv >= depth or len(rows) < 2 or (lv > 0 and rng.random() < leaf_prob):
                continue
            f = int(rng.choice(feats))
            side = 1.0 if rng.random() < 0.5 else -1.0
            q = 0.5 + side * skew * 0.45
            x = Xd[rows, f]
            t = float(np.quantile(x, q))
            lm = x <= t
            if lm.all() or (~lm).all():
                continue
            feat[nid] = f; thr[nid] = t
            l, r = new_node(), new_node()
            left[nid] = l; right[nid] = r
            nxt.append((l, rows[lm], lv + 1)); nxt.append((r, rows[~lm], lv + 1))
        frontier = nxt
    return (np.array(feat), np.array(thr), np.array(left), np.array(right), np.array(value))


def _eval_teacher(T, X):
    feat, thr, left, right, value = T
    node = np.zeros(len(X), dtype=np.int64)
    while True:
        f = feat[node]
        inner = f >= 0
        if not inner.any():
            break
        idx = np.nonzero(inner)[0]
        go_left = X[idx, f[idx]] <= thr[node[idx]]
        node[idx] = np.where(go_left, left[node[idx]], right[node[idx]])
    return value[node]


def _build_function(p, plan, rng_f, n_out):
    Xd = _gen_rows(p, plan, rng_f, DESIGN_N)
    feats = plan["informative"]
    funcs = []
    for _ in range(n_out):
        teachers = [_build_teacher(Xd, feats, rng_f, p["teacher_depth"], p["skew"], p["leaf_prob"])
                    for _ in range(p["n_teachers"])]
        a = rng_f.standard_normal(len(feats)); w = rng_f.uniform(0.5, 2.0, len(feats))
        st = sum(_eval_teacher(T, Xd) for T in teachers)
        ss = (np.sin(Xd[:, feats] * w) * a).sum(1)
        norm = (st.mean(), st.std() + 1e-12, ss.mean(), ss.std() + 1e-12)
        funcs.append((teachers, a, w, norm))
    return funcs, Xd


def _eval_function(fn, p, plan, X):
    teachers, a, w, (m1, s1, m2, s2) = fn
    feats = plan["informative"]
    st = (sum(_eval_teacher(T, X) for T in teachers) - m1) / s1
    ss = ((np.sin(X[:, feats] * w) * a).sum(1) - m2) / s2
    return (1 - p["alpha"]) * st + p["alpha"] * ss


def generate(**kw):
    p = dict(DEFAULTS)
    unknown = set(kw) - set(p)
    if unknown:
        raise ValueError(f"unknown synth knobs {unknown}")
    p.update(kw)
    seed = int(p["seed"])
    rng_f = np.random.default_rng([seed, 1])
    plan = _feature_plan(p, rng_f)
    K = p["n_classes"] if p["task"] == "multiclass" else 1
    funcs, Xd = _build_function(p, plan, rng_f, K)
    gsz = None
    if p["n_groups"] > 0:
        G = p["n_groups"]
        gw = 1.0 / np.arange(1, G + 1) ** p["group_skew"]
        gcw = np.cumsum(gw / gw.sum())
        geff = rng_f.standard_normal((G, K)) * p["group_effect"]
    fd = np.stack([_eval_function(fn, p, plan, Xd) for fn in funcs], 1)
    thr_bin = float(np.quantile(fd[:, 0], 1 - p["pos_rate"])) if p["task"] == "binary" else None
    n = int(p["n"])
    Xs, ys, gs = [], [], []
    for ci in range((n + CHUNK - 1) // CHUNK):
        m = min(CHUNK, n - ci * CHUNK)
        r_x = np.random.default_rng([seed, 2, ci])
        r_n = np.random.default_rng([seed, 3, ci])
        r_m = np.random.default_rng([seed, 4, ci])
        r_g = np.random.default_rng([seed, 5, ci])
        X = _gen_rows(p, plan, r_x, CHUNK)[:m]
        F = np.stack([_eval_function(fn, p, plan, X) for fn in funcs], 1)
        g = None
        if p["n_groups"] > 0:
            g = np.minimum(np.searchsorted(gcw, r_g.random(CHUNK)[:m], side="right"), p["n_groups"] - 1)
            F = F + geff[g]
        if p["task"] == "regression":
            y = F[:, 0] + p["noise"] * r_n.standard_normal(CHUNK)[:m]
        elif p["task"] == "binary":
            y = (F[:, 0] > thr_bin).astype(np.int64)
            flip = r_n.random(CHUNK)[:m] < p["noise"]
            y = np.where(flip, 1 - y, y)
        else:
            y = F.argmax(1)
            flip = r_n.random(CHUNK)[:m] < p["noise"]
            other = (y + 1 + r_n.integers(0, K - 1, CHUNK)[:m]) % K
            y = np.where(flip, other, y)
        if p["missing_rate"] > 0:
            X = X.copy()
            X[r_m.random((CHUNK, p["d"]))[:m] < p["missing_rate"]] = np.nan
        Xs.append(X); ys.append(y); gs.append(g)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    g = np.concatenate(gs) if p["n_groups"] > 0 else None
    cols = {}
    for j in range(p["d"]):
        k = plan["kinds"][j]
        if k == "cat":
            K_ = int(plan["card"][j])
            lab = [f"c{i}" for i in range(K_)]
            codes = np.where(np.isnan(X[:, j]), -1, np.nan_to_num(X[:, j], nan=0)).astype(np.int64)
            cols[f"f{j}_cat"] = pd.Categorical.from_codes(codes, categories=lab)
        else:
            cols[f"f{j}_{k}{'_inf' if j in set(plan['informative']) else ''}"] = X[:, j]
    df = pd.DataFrame(cols)
    info = {"generator": "prep/synth.py", "params": p, "informative_features": plan["informative"].tolist(),
            "feature_kinds": plan["kinds"].tolist()}
    return df, y, g, info


def prepare(name, out_dir, cache=None, **overrides):
    import io_utils as U
    kw = dict(INSTANCES.get(name, {}))
    kw.update(overrides)
    df, y, g, info = generate(**kw)
    task = info["params"]["task"]
    enc = U.encode_frame(df)
    classes = None if task == "regression" else [str(i) for i in range(int(y.max()) + 1)]
    if task == "binary":
        classes = ["0", "1"]
    group = gnames = gcounts = None
    if g is not None:
        labels = [f"g{i}" for i in range(info["params"]["n_groups"])]
        group, gnames, gcounts = U.group_ids(g, labels)
    return U.write_dataset(out_dir, name, enc, y, task, classes, info, group=group, group_names=gnames,
                           group_counts=gcounts, group_source=("synthetic group id" if g is not None else None),
                           extra_meta={"group_set": "synthetic"}, outer="mask")


def parse_knobs(s):
    out = {}
    for kv in filter(None, (s or "").split(",")):
        k, v = kv.split("=")
        k = k.strip()
        typ = type(DEFAULTS[k])
        out[k] = v if typ is str else (int(float(v)) if typ is int else float(v))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--knobs", default="", help="comma list, e.g. n=1000000,d=64,skew=0.9,seed=1")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    m = prepare(a.name, Path(a.out), **parse_knobs(a.knobs))
    print(json.dumps({k: m[k] for k in ("name", "n", "d", "bytes", "sha256", "content_sha256")}))
