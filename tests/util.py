import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from mangrove import ModelSpec


def synth(n, d, nb=16, seed=0, task="binary", K=3):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(0, nb, (n, d)).astype(np.uint8)
    z = Xb[:, 0].astype(float) * 0.3 - Xb[:, 1] * 0.2 + np.sin(Xb[:, 2 % d].astype(float)) + rng.normal(0, 1.0, n)
    if task == "binary":
        y = (z > np.median(z)).astype(np.float32)
    elif task == "regression":
        y = z.astype(np.float32)
    else:
        y = np.digitize(z, np.quantile(z, np.linspace(0, 1, K + 1)[1:-1])).astype(np.int64)
    folds = rng.integers(0, 5, n)
    return Xb, y, folds


def sample_hpo(n_cfg, seed, depth="mixed", dmin=3, dmax=10, sub=None, col=None, bits=(16,), esr=0):
    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n_cfg))
    eta, lam, mcw = lu(0.02, 0.3), lu(0.1, 10.0), lu(0.5, 20.0)
    md = np.full(n_cfg, 6) if depth == "fixed6" else rng.integers(dmin, dmax + 1, n_cfg)
    out = []
    for i in range(n_cfg):
        cfg = dict(eta=float(eta[i]), lam=float(lam[i]), mcw=float(mcw[i]), max_depth=int(md[i]), quant_bits=int(rng.choice(bits)))
        if sub is not None:
            cfg["subsample"] = float(rng.uniform(*sub))
        if col is not None:
            cfg["colsample_bytree"] = float(rng.uniform(*col))
            cfg["colsample_bylevel"] = float(rng.uniform(*col))
        out.append(cfg)
    return out


def make_specs(cfgs, folds, data_key="g", n_rounds=10, seed_base=1000, base_margin=0.0, fold_list=None, **kw):
    specs = []
    rows = np.arange(len(folds))
    fold_list = range(5) if fold_list is None else fold_list
    i = 0
    for cfg in cfgs:
        for k in fold_list:
            specs.append(ModelSpec(data_key, rows[folds != k], rows[folds == k], n_rounds=n_rounds, seed=seed_base + i,
                                   base_margin=base_margin, tag=(cfg, k), **cfg, **kw))
            i += 1
    return specs


def tree_sig(trees):
    parts = []
    for rt in trees:
        for t in rt:
            parts.append((t.split_node.tobytes(), t.split_feat.tobytes(), t.split_thr.astype(np.int32).tobytes(),
                          t.leaf_node.tobytes(), t.leaf_val.view(np.int32).tobytes()))
    return tuple(parts)


def obs(eng, mid, rounds=None):
    st = eng.reg[mid]
    q = st.local_round if rounds is None else min(rounds, st.local_round)
    parts = [q, tree_sig(st.trees[:q]), tuple(st.metric_q_history[:q]),
             np.asarray(st.metric_history[:q], dtype=np.float64).tobytes()]
    if rounds is None:
        intrinsic = "active"
        if q >= st.spec.n_rounds:
            intrinsic = "done:n_rounds"
        elif (st.metric_history and st.spec.early_stopping_rounds > 0
              and q - 1 - st.best_round >= st.spec.early_stopping_rounds):
            intrinsic = "done:early_stop"
        parts += [intrinsic, st.best_round, st.best_metric]
        if len(st.spec.eval_rows):
            parts.append(eng.eval_margin(mid).view(np.int32).tobytes())
    return tuple(parts)
