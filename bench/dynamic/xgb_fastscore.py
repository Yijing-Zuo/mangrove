import os

import numpy as np
import torch

from .. import metrics as M
from .mg_fastscore import (_EXACT_LIMIT, WARMUP_N, _ContentCache, _Labels, _logloss, _logloss_buffered, rank_sum2,
                           same_metrics)

MODES = ("full", "auc")


def _plain_logloss(lab, margin):
    return _logloss(lab.y, margin)


class WorkerScorer:
    def __init__(self, ds, auc_device, mode="full", check=None):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if check is None:
            check = os.environ.get("MANGROVE_BENCH_XGB_FASTSCORE_CHECK", "0") == "1"
        self.ds, self.task, self.mode, self.check = ds, ds.task, mode, bool(check)
        self.auc_device = None
        if auc_device is not None and self.task == "binary":
            dev = torch.device(auc_device)
            if dev.type != "cuda" or torch.cuda.is_available():
                self.auc_device = dev
        self._ll = _logloss_buffered if mode == "full" else _plain_logloss
        self._labels = _ContentCache()
        self.counters = dict(rank_models=0, host_models=0, checked=0)

    def labels(self, j):
        return self._labels.get(int(j.part), (j.val_mask,),
                                lambda: _Labels(self.ds.y[j.val_mask], self.task == "binary"))

    def _fast(self, lab, m):
        if self.auc_device is None or m.dtype != np.float32 or m.ndim != 1:
            return False
        n = m.shape[0]
        return n == len(lab.y) and n > 0 and n * (n + 1) // 2 < _EXACT_LIMIT and bool(np.isfinite(m).all())

    def _score(self, lab, m):
        if not self._fast(lab, m):
            return M.metrics(self.task, lab.y, m), "host"
        ll = self._ll(lab, m)
        if lab.npos == 0 or lab.nneg == 0:
            auc = float("nan")
        else:
            s2 = rank_sum2([(m, lab.pos_dev(self.auc_device))], self.auc_device)[0]
            auc = float((np.float64(s2) / 2.0 - lab.npos * (lab.npos + 1) / 2) / (lab.npos * lab.nneg))
        return {"auc": auc, "logloss": ll}, "rank"

    def metrics(self, j, margin):
        m = np.asarray(margin)
        out, path = self._score(self.labels(j), m)
        self.counters[path + "_models"] += 1
        if self.check:
            ref = M.metrics(self.task, self.ds.y[j.val_mask], m)
            if not same_metrics(out, ref):
                raise RuntimeError(f"MANGROVE_BENCH_XGB_FASTSCORE_CHECK: model cfg={getattr(j, 'cfg', None)} part={j.part}: "
                                   f"fast {out} != M.metrics {ref}")
            self.counters["checked"] += 1
        return out, path

    def warmup(self):
        y = (np.arange(WARMUP_N) % 2).astype(self.ds.y.dtype)
        m = np.linspace(-1.0, 1.0, WARMUP_N, dtype=np.float32)
        m[::3] = 0.0
        self._ll(_Labels(y[:40], True), m[:40])
        if self.auc_device is not None:
            self._score(_Labels(y, True), m)
        for k in self.counters:
            self.counters[k] = 0

    def describe(self):
        return {"mode": self.mode, "auc_device": None if self.auc_device is None else str(self.auc_device),
                "gpu_auc": self.auc_device is not None and self.auc_device.type == "cuda", "check": self.check}
