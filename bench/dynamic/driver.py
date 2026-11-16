import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os

_FOLD_SUM = os.environ.get("MANGROVE_BENCH_DRIVER_FOLD_SUM", "").strip() or "fold"
if _FOLD_SUM not in ("fold", "completion"):
    raise ValueError(f"MANGROVE_BENCH_DRIVER_FOLD_SUM={_FOLD_SUM!r}: expected fold or completion")
FOLD_SUM_COMPLETION = _FOLD_SUM == "completion"


@dataclass
class Segment:
    seg_id: int
    trial: int
    job: Any
    start: int
    end: int
    token: Any = None


@dataclass
class Done:
    seg: Segment
    loss: float
    metrics: Dict[str, float]
    token: Any
    rounds_trained: int
    busy_s: float = 0.0


def rungs(r_min, r_max, eta):
    out, r = [], r_min
    while r < r_max:
        out.append(int(r))
        r *= eta
    out.append(int(r_max))
    return out


@dataclass
class Trial:
    tid: int
    jobs: List[Any]
    rung_scores: Dict[int, float] = field(default_factory=dict)
    fold_loss: Dict[int, Dict[int, float]] = field(default_factory=dict)
    tokens: Dict[int, Any] = field(default_factory=dict)
    rounds: Dict[int, int] = field(default_factory=dict)
    level: int = -1
    promoted_from: set = field(default_factory=set)
    in_flight: int = 0
    stopped: bool = False


class Driver:
    def __init__(self, trainer, jobs_by_cfg, task, dyn, clock=time.perf_counter):
        self.tr = trainer
        self.jobs_by_cfg = jobs_by_cfg
        self.task = task
        self.dyn = dict(dyn)
        self.algo = dyn["algo"]
        self.variant = dyn.get("variant", "promote" if self.algo == "asha" else "sync")
        self.eta = int(dyn.get("eta", 3))
        self.r_min = int(dyn.get("r_min", 10))
        self.r_max = int(dyn.get("r_max", 270))
        self.max_trials = dyn.get("n_trials")
        self.clock = clock
        self.trials: Dict[int, Trial] = {}
        self.trace = []
        self.seg_id = 0
        self.mr_exec = 0
        self.pending: List[Segment] = []
        self.wall0 = None
        if not trainer.supports_resume and self.algo == "asha":
            raise ValueError("ASHA needs a trainer that can resume models (continuation)")
        if self.algo == "asha" and self.variant == "promote" and not getattr(trainer, "supports_pause", True):
            raise ValueError("this trainer cannot pause resident models: use dynamic.variant='stop'")

    def _now(self):
        return self.clock() - self.wall0

    def _new_trial(self, cfg):
        t = Trial(cfg, list(self.jobs_by_cfg[cfg]))
        for j in t.jobs:
            t.rounds[j.part] = 0
        self.trials[cfg] = t
        return t

    def _segments_to(self, t, target):
        segs = []
        for j in t.jobs:
            self.seg_id += 1
            start = t.rounds[j.part] if self.tr.supports_resume else 0
            segs.append(Segment(self.seg_id, t.tid, j, start, target, t.tokens.get(j.part)))
        t.in_flight += len(segs)
        return segs

    def _flush(self):
        free = self.tr.capacity() - self.tr.in_flight()
        while self.pending and free > 0:
            self.tr.submit(self.pending.pop(0))
            free -= 1

    def _incumbents(self):
        best_any = min((s for t in self.trials.values() for s in t.rung_scores.values()), default=None)
        top = len(self.rung_list) - 1
        best_max = min((t.rung_scores[top] for t in self.trials.values() if top in t.rung_scores), default=None)
        return best_any, best_max

    def _record(self, t, k):
        best_any, best_max = self._incumbents()
        self.trace.append({"t": self._now(), "trial": t.tid, "rung": k, "rounds": self.rung_list[k],
                           "score": t.rung_scores[k], "incumbent_any": best_any, "incumbent_max_rung": best_max,
                           "model_rounds_executed": self.mr_exec, "utilization": self.tr.utilization_now(),
                           "metrics_mean": getattr(t, "rung_metrics", {}).get(k),
                           "fold_scores": dict(t.fold_loss[k]),
                           "fold_metrics": dict(t.fold_metrics[k])})

    def _absorb(self, d: Done):
        t = self.trials[d.seg.trial]
        j = d.seg.job
        t.in_flight -= 1
        t.tokens[j.part] = d.token
        t.rounds[j.part] = d.seg.end
        self.mr_exec += d.rounds_trained
        k = self.rung_list.index(d.seg.end)
        t.fold_loss.setdefault(k, {})[j.part] = d.loss
        t.last_metrics = getattr(t, "last_metrics", {})
        t.last_metrics[j.part] = (d.seg.end, d.metrics)
        t.fold_metrics = getattr(t, "fold_metrics", {})
        t.fold_metrics.setdefault(k, {})[j.part] = d.metrics or {}
        if len(t.fold_loss[k]) == len(t.jobs):
            parts = list(t.fold_loss[k]) if FOLD_SUM_COMPLETION else sorted(t.fold_loss[k])
            t.rung_scores[k] = float(sum(t.fold_loss[k][p] for p in parts) / len(t.jobs))
            t.level = k
            fm = [t.fold_metrics[k][p] for p in (list(t.fold_metrics[k]) if FOLD_SUM_COMPLETION
                                                 else sorted(t.fold_metrics[k]))]
            keys = set.intersection(*[set(m) for m in fm]) if fm else set()
            t.rung_metrics = getattr(t, "rung_metrics", {})
            t.rung_metrics[k] = {m: float(sum(x[m] for x in fm) / len(t.jobs)) for m in keys
                                 if all(isinstance(x[m], (int, float)) and x[m] == x[m] for x in fm)}
            self._record(t, k)
            return t, k
        return None, None

    def _rung_members(self, k):
        return sorted((t.rung_scores[k], t.tid) for t in self.trials.values() if k in t.rung_scores)

    def _wait(self):
        done = self.tr.wait()
        out = []
        for d in done:
            t, k = self._absorb(d)
            if t is not None:
                out.append((t, k))
        return out

    def run(self):
        self.rung_list = rungs(self.r_min, self.r_max, self.eta)
        self.wall0 = self.clock()
        cfgs = list(self.jobs_by_cfg)
        if self.max_trials:
            cfgs = cfgs[: int(self.max_trials)]
        if hasattr(self.tr, "clock0"):
            self.tr.clock0 = self.wall0
        if self.algo == "grid":
            self.rung_list = [self.r_max]
            self._sha(cfgs, 0)
        elif self.algo == "sha":
            self._sha(cfgs, 0)
        elif self.algo == "hyperband":
            self._hyperband(cfgs)
        elif self.algo == "hbpar":
            self._hbpar(cfgs)
        elif self.variant == "stop":
            self._asha_stop(cfgs)
        else:
            self._asha_promote(cfgs)
        wall = self._now()
        occ = self.tr.occupancy() if hasattr(self.tr, "occupancy") else None
        per_model = []
        for t in self.trials.values():
            for j in t.jobs:
                rounds, metrics = t.last_metrics[j.part]
                per_model.append({"mid": j.mid, "cfg": j.cfg, "part": j.part, "seed": j.seed,
                                  "rounds": rounds, "metrics": metrics})
        return {"wall_s": wall, "trace": self.trace, "model_rounds_executed": self.mr_exec, "rungs": self.rung_list,
                "per_model": per_model,
                "n_trials": len(self.trials), "trainer": self.tr.stats(wall), "occupancy": occ,
                "brackets": getattr(self, "bracket_log", None)}

    def _sha(self, cfgs, k0):
        alive = [self._new_trial(c) for c in cfgs]
        for k in range(k0, len(self.rung_list)):
            for t in alive:
                self.pending.extend(self._segments_to(t, self.rung_list[k]))
            while any(t.in_flight for t in alive) or self.pending:
                self._flush()
                self._wait()
            if k == len(self.rung_list) - 1:
                break
            ranked = sorted(alive, key=lambda t: (t.rung_scores[k], t.tid))
            keep = max(1, len(ranked) // self.eta)
            for t in ranked[keep:]:
                t.stopped = True
                for tok in t.tokens.values():
                    self.tr.release(tok)
            alive = ranked[:keep]

    def _brackets(self, cfgs):
        R, eta = self.r_max / self.r_min, self.eta
        s_max = int(math.floor(math.log(R, eta) + 1e-9))
        out, nxt, it = [], 0, 0
        while nxt < len(cfgs):
            for s in range(s_max, -1, -1):
                n = int(math.ceil((s_max + 1) / (s + 1) * eta ** s))
                take = cfgs[nxt:nxt + n]
                nxt += n
                if take:
                    out.append((s, s_max - s, take, it))
                if nxt >= len(cfgs):
                    break
            it += 1
            if self.dyn.get("hb_iterations") and it >= int(self.dyn["hb_iterations"]):
                break
        self.bracket_log = [{"s": s, "k0": k0, "n": len(take), "iteration": i} for s, k0, take, i in out]
        return out

    def _hyperband(self, cfgs):
        for s, k0, take, _ in self._brackets(cfgs):
            self._sha(take, k0)

    def _hbpar(self, cfgs):
        B = []
        for s, k0, take, _ in self._brackets(cfgs):
            alive = [self._new_trial(c) for c in take]
            b = {"k": k0, "alive": alive}
            for t in alive:
                t.bracket = len(B)
                self.pending.extend(self._segments_to(t, self.rung_list[k0]))
            B.append(b)
        top = len(self.rung_list) - 1
        while self.pending or self.tr.in_flight():
            self._flush()
            for t, k in self._wait():
                b = B[t.bracket]
                if k != b["k"] or any(k not in x.rung_scores for x in b["alive"]):
                    continue
                if k == top:
                    for x in b["alive"]:
                        for tok in x.tokens.values():
                            self.tr.release(tok)
                    b["alive"] = []
                    continue
                ranked = sorted(b["alive"], key=lambda x: (x.rung_scores[k], x.tid))
                keep = max(1, len(ranked) // self.eta)
                for x in ranked[keep:]:
                    x.stopped = True
                    for tok in x.tokens.values():
                        self.tr.release(tok)
                b["alive"], b["k"] = ranked[:keep], k + 1
                cont = []
                for x in b["alive"]:
                    cont.extend(self._segments_to(x, self.rung_list[k + 1]))
                n_cont = 0
                while n_cont < len(self.pending) and self.trials[self.pending[n_cont].trial].level >= 0:
                    n_cont += 1
                self.pending[n_cont:n_cont] = cont

    def _asha_promote(self, cfgs):
        queue = list(cfgs)
        top = len(self.rung_list) - 1

        def next_job():
            for k in range(top - 1, -1, -1):
                mem = self._rung_members(k)
                n_top = len(mem) // self.eta
                for score, tid in mem[:n_top]:
                    t = self.trials[tid]
                    if k not in t.promoted_from and t.level == k and t.in_flight == 0:
                        t.promoted_from.add(k)
                        return self._segments_to(t, self.rung_list[k + 1])
            if queue:
                return self._segments_to(self._new_trial(queue.pop(0)), self.rung_list[0])
            return None
        while True:
            while self.tr.capacity() - self.tr.in_flight() - len(self.pending) > 0:
                segs = next_job()
                if segs is None:
                    break
                self.pending.extend(segs)
            self._flush()
            if self.tr.in_flight() == 0 and not self.pending:
                if next_job_possible := next_job():
                    self.pending.extend(next_job_possible)
                    continue
                break
            self._wait()
        for t in self.trials.values():
            for tok in t.tokens.values():
                self.tr.release(tok)

    def _asha_stop(self, cfgs):
        queue = list(cfgs)
        top = len(self.rung_list) - 1
        while True:
            while queue and self.tr.capacity() - self.tr.in_flight() - len(self.pending) > 0:
                self.pending.extend(self._segments_to(self._new_trial(queue.pop(0)), self.rung_list[0]))
            self._flush()
            if self.tr.in_flight() == 0 and not self.pending:
                break
            for t, k in self._wait():
                if k == top:
                    for tok in t.tokens.values():
                        self.tr.release(tok)
                    continue
                mem = [s for s, _ in self._rung_members(k)]
                idx = max(0, len(mem) // self.eta - 1)
                if t.rung_scores[k] <= mem[idx]:
                    self.pending[:0] = self._segments_to(t, self.rung_list[k + 1])
                else:
                    t.stopped = True
                    for tok in t.tokens.values():
                        self.tr.release(tok)
