import hashlib
import numpy as np

PRIMARY = {"binary": ("auc", +1), "regression": ("rmse", -1), "multiclass": ("mlogloss", -1)}
LOSS_METRIC = {"binary": "logloss", "regression": "rmse", "multiclass": "mlogloss"}


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    _, idx, cnt = np.unique(ss, return_index=True, return_counts=True)
    ranks[order] = np.repeat(idx + (cnt + 1) / 2.0, cnt)
    npos = float(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(task, y, margin):
    margin = np.asarray(margin, dtype=np.float64)
    y = np.asarray(y)
    if task == "binary":
        p = np.clip(1.0 / (1.0 + np.exp(-margin)), 1e-15, 1 - 1e-15)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return {"auc": auc(y, margin), "logloss": ll}
    if task == "regression":
        return {"rmse": float(np.sqrt(np.mean((margin - y) ** 2)))}
    z = margin - margin.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    yi = y.astype(np.int64)
    ll = float(-np.mean(np.log(np.clip(p[np.arange(len(yi)), yi], 1e-15, None))))
    return {"mlogloss": ll, "acc": float(np.mean(p.argmax(1) == yi))}


def loss_value(task, m):
    return m[LOSS_METRIC[task]]


def margin_hash(margin):
    a = np.ascontiguousarray(np.asarray(margin, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]
