import argparse
import copy
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
import numpy as np
import torch
from mangrove import Engine, Cohort, ModelSpec

BUDGETS = {'cpu': (2000000, 3000000, 6000000), 'gpu': (900000, 1200000, 2300000)}
SMALL_BUDGETS = (100000, 120000, 140000)
PROCEDURES = {'grid': 'random', 'sha': 'sha', 'asha': 'asha', 'hbpar': 'hyperband-parallel', 'hyperband': 'hyperband-sequential'}


def table(loss, n=3000, d=8, bins=32):
    rng = np.random.default_rng(0)
    X = rng.integers(0, bins, (n, d)).astype(np.uint8)
    z = X[:, 0] * 0.3 - X[:, 1] * 0.2 + np.sin(X[:, 2].astype(float)) + rng.normal(0, 1, n)
    y = (z > np.median(z)).astype(np.float32) if loss == 'logistic' else z.astype(np.float32)
    return [Cohort('t', X, y)], rng.integers(0, 5, n)


def models(folds, sampling, configs=6, rounds=6):
    rng = np.random.default_rng(1)
    rows = np.arange(len(folds))
    specs = []
    for c in range(configs):
        p = dict(eta=float(np.exp(rng.uniform(np.log(.05), np.log(.3)))), lam=float(np.exp(rng.uniform(np.log(.1), np.log(5)))),
                 mcw=float(rng.uniform(0, 3)), max_depth=int(rng.integers(2, 8)))
        if sampling:
            p.update(subsample=float(rng.uniform(.5, 1)), colsample_bytree=float(rng.uniform(.5, 1)),
                     colsample_bylevel=float(rng.uniform(.5, 1)))
        for k in range(3):
            specs.append(ModelSpec('t', rows[folds != k], rows[folds == k], n_rounds=rounds, seed=1000 + len(specs), **p))
    return specs


def small_models():
    rng = np.random.default_rng(4242)
    X = rng.integers(0, 16, (800, 6)).astype(np.uint8)
    y = (X[:, 0] * 0.3 - X[:, 1] * 0.2 + rng.normal(0, 1, 800)).astype(np.float32)
    pick = np.random.default_rng(99)
    specs = []
    for i in range(4):
        train = np.sort(pick.permutation(800)[:64])
        specs.append(ModelSpec('t', train, np.setdiff1d(np.arange(800), train)[:100], n_rounds=3, max_depth=7, mcw=0., lam=.01,
                               seed=11 + i, acc_mode='wide', subsample=.8, colsample_bytree=.5))
    return [Cohort('t', X, y)], specs


def fingerprint(pairs):
    h = hashlib.sha256()
    for e, m in pairs:
        s = e.reg[m]
        for trees in s.trees:
            for t in trees:
                for f in ('split_node', 'split_feat', 'split_thr', 'leaf_node', 'leaf_val'):
                    h.update(np.ascontiguousarray(getattr(t, f)).tobytes())
        h.update(np.ascontiguousarray(e.eval_margin(m), dtype=np.float32).tobytes())
        h.update(json.dumps([s.local_round, list(s.metric_q_history), [float(x) for x in s.metric_history]]).encode())
    return h.hexdigest()[:16]


def train(cohorts, loss, specs, kw, arm, budget=2 ** 40):
    if arm == 'standalone':
        pairs = []
        for s in specs:
            e = Engine(cohorts, loss, **kw)
            pairs.append((e, e.add_models([copy.copy(s)])[0]))
            e.run()
        return pairs, None
    if arm == 'lifecycle':
        e = Engine(cohorts, loss, budget_bytes=budget, max_model_slots=8, **kw)
        half = len(specs) // 2
        ids = e.add_models([copy.copy(s) for s in specs[:half]])
        e.step()
        e.step()
        paused = e.active_ids()[:2]
        e.pause(paused)
        e.step()
        e.resume(paused)
        e.compact(order=e.active_ids()[::-1])
        ids += e.add_models([copy.copy(s) for s in specs[half:]])
    else:
        e = Engine(cohorts, loss, budget_bytes=budget, chunk_parents=1 if arm == 'chunked' else None, **kw)
        ids = e.add_models([copy.copy(s) for s in specs])
    e.run()
    events = [x['event'] for x in e.events]
    return [(e, m) for m in ids], dict(chunked=sum(c['chunk'] < c['n_groups'] for c in e.counters), levels=len(e.counters),
                                       pauses=events.count('pause'), evictions=events.count('evict'))


def quick(device):
    if device == 'gpu' and not torch.cuda.is_available():
        sys.exit('CUDA is not available')
    if device == 'gpu':
        os.environ['MANGROVE_NODEMAP_REQUIRE'] = '1'
    kw = dict(device='cuda', backend='gpu') if device == 'gpu' else dict(device='cpu', backend='ref')
    t0 = time.time()
    same, total = True, dict(chunked=0, pauses=0, evictions=0)
    blocks = [(loss, sampling) for loss in ('squared', 'logistic') for sampling in (False, True)] + [('squared', None)]
    for loss, sampling in blocks:
        if sampling is None:
            cohorts, specs = small_models()
            arms = [('standalone', None), ('fused', 2 ** 40)] + [('fused', b) for b in SMALL_BUDGETS]
            name = f'{loss:8s} small      '
        else:
            cohorts, folds = table(loss)
            specs = models(folds, sampling)
            arms = [('standalone', None), ('fused', 2 ** 40), ('chunked', 2 ** 40)] + [('lifecycle', b) for b in BUDGETS[device]]
            name = f"{loss:8s} {'sampling' if sampling else 'no sampling':11s}"
        seen = set()
        for arm, budget in arms:
            pairs, c = train(cohorts, loss, specs, kw, arm, budget)
            f = fingerprint(pairs)
            seen.add(f)
            label = arm if budget in (None, 2 ** 40) else f'{arm} {budget / 1e6:g} MB'
            info = ''
            if c is not None:
                for k in total:
                    total[k] += c[k]
                info = f"  {c['chunked']} of {c['levels']} levels in chunks, {c['pauses']} pauses, {c['evictions']} evictions"
            print(f'{name} {label:20s} {f}{info}', flush=True)
        same &= len(seen) == 1
    exercised = all(v > 0 for v in total.values())
    print(f"{total['chunked']} levels in chunks, {total['pauses']} pauses, {total['evictions']} evictions, {time.time() - t0:.1f} s")
    print('identical' if same and exercised else 'DIFFERENT' if not same else 'identical, but chunking, pauses or evictions did not occur')
    sys.exit(0 if same and exercised else 1)


def same_metrics(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
        return False
    for k in a:
        x, y = float(a[k]), float(b[k])
        if not (struct.pack('<d', x) == struct.pack('<d', y) or (math.isnan(x) and math.isnan(y))):
            return False
    return True


def tree_signature(st):
    h = hashlib.sha256()
    for tr in st.trees:
        for t in (tr if isinstance(tr, (list, tuple)) else [tr]):
            for f in ('split_node', 'split_feat', 'split_thr', 'leaf_node', 'leaf_val'):
                h.update(np.ascontiguousarray(getattr(t, f)).tobytes())
    return h.hexdigest()


def sample_models(per_model, k):
    by_r = {}
    for p in per_model:
        by_r.setdefault(int(p['rounds']), []).append(p)
    out = []
    for r in sorted(by_r, reverse=True):
        pool = sorted(by_r[r], key=lambda p: (p['cfg'], p['part']))
        need = k - len(out)
        if need <= 0:
            break
        if len(pool) <= need:
            out += pool
        else:
            out += [pool[(i * len(pool)) // need] for i in range(need)]
    return out


def retrain_alone(path, k, out_dir, mg, mods):
    M, runner, build_jobs, config_list_hash, WorkloadSpec, prepare_dataset, FastScorer = mods
    t0 = time.perf_counter()
    r = json.load(open(path, encoding='utf-8'))
    wl = dict(r['workload'])
    fields = set(WorkloadSpec.__dataclass_fields__)
    spec = WorkloadSpec(**{key: v for key, v in wl.items() if key in fields})
    ds = prepare_dataset(spec, None)
    jobs = build_jobs(spec, ds)
    by_key = {(j.cfg, j.part): j for j in jobs}
    hash_all = config_list_hash(jobs)
    r_max = int((spec.dynamic or {}).get('r_max') or spec.rounds)
    tr = (r.get('dynamic_summary') or {}).get('trainer') or {}
    ek = dict(tr.get('engine_kwargs') or {})
    if not ek:
        raise SystemExit(f'{path}: no dynamic_summary.trainer.engine_kwargs (not a Mangrove search result?)')
    scorer = FastScorer(ds, auc_device=None)
    cohort_ids = sorted({int(c) for c in ([-1] if spec.cuts == 'global' else range(ds.n_folds))})
    cohorts = runner.make_cohorts(mg, ds, cohort_ids)
    sample = sample_models(r['per_model'], k)
    rows = []
    for p in sample:
        j = by_key[(p['cfg'], p['part'])]
        if int(j.seed) != int(p['seed']) or ('mid' in p and int(j.mid) != int(p['mid'])):
            raise SystemExit(f"{path}: job (cfg {p['cfg']}, part {p['part']}) seed/mid {j.seed}/{j.mid} != "
                             f"per_model {p['seed']}/{p.get('mid')}: spec reconstruction does not match the run")
        train_rows, eval_rows, base = scorer.spec_rows(j)
        kw = dict(data_key=int(j.cohort), train_rows=train_rows, eval_rows=eval_rows,
                  eta=j.params['eta'], lam=j.params['lam'], mcw=j.params['mcw'], max_depth=j.params['max_depth'],
                  n_rounds=r_max, subsample=j.params['subsample'], colsample_bytree=j.params['colsample'],
                  seed=j.seed, base_margin=base, tag=(j.cfg, j.part))
        sp = mg.ModelSpec(**kw)
        ta = time.perf_counter()
        eng = mg.Engine(cohorts, runner.LOSS[ds.task], n_classes=ds.K, **ek)
        mid = eng.add_models([sp])[0]
        target = int(p['rounds'])
        steps = 0
        while eng.reg[mid].local_round < target:
            eng.step()
            steps += 1
            if steps > 10 * r_max:
                raise RuntimeError('alone run made no progress')
        eng.nm_check()
        st = eng.reg[mid]
        em = np.asarray(eng.eval_margin(mid))
        alone = M.metrics(ds.task, ds.y[j.val_mask], em)
        torch.cuda.synchronize()
        rows.append(dict(cfg=p['cfg'], part=p['part'], mid=p.get('mid'), seed=p['seed'], rounds=target,
                         local_round=int(st.local_round), status=st.status, batched=p['metrics'],
                         alone={key: float(v) for key, v in alone.items()}, equal=same_metrics(p['metrics'], alone),
                         tree_sig=tree_signature(st), n_trees=len(st.trees),
                         margin_sha=hashlib.sha256(np.ascontiguousarray(em.astype(np.float32)).tobytes()).hexdigest(),
                         nm_rounds=int(eng.nm_stats.get('rounds', 0)), alone_s=round(time.perf_counter() - ta, 3)))
        del eng
    scorer.close()
    rep = dict(input=str(path), input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), dataset=spec.dataset,
               algo=(spec.dynamic or {}).get('algo'), n_configs=spec.n_configs, r_max=r_max,
               config_list_hash_result=r.get('config_list_hash'), config_list_hash_rebuilt=hash_all, k=k, n_checked=len(rows),
               all_equal=bool(rows) and all(x['equal'] for x in rows), models=rows, wall_s=round(time.perf_counter() - t0, 2))
    out = Path(out_dir) / f'alone_{Path(path).parent.name}_{Path(path).stem}.json'
    out.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    return rep, out


def alone(results, k, out_dir):
    results = [str(Path(p).resolve()) for p in results]
    out_dir = Path(out_dir).resolve()
    os.environ.setdefault('MANGROVE_NODEMAP_REQUIRE', '1')
    out_dir.mkdir(parents=True, exist_ok=True)
    from bench import metrics as M
    from bench.runners import mangrove as runner
    from bench.workload import build_jobs, config_list_hash, WorkloadSpec
    from bench.data import prepare_dataset
    from bench.dynamic.mg_fastscore import FastScorer
    from bench import envinfo
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(envinfo.MEM_FRACTION, 0)
    mg = runner.import_mangrove({})
    mods = (M, runner, build_jobs, config_list_hash, WorkloadSpec, prepare_dataset, FastScorer)
    summary, ok = [], True
    for p in results:
        try:
            rep, out = retrain_alone(p, k, out_dir, mg, mods)
        except SystemExit as e:
            print(f'reconstruction error {p}: {e}', flush=True)
            summary.append(dict(input=str(p), error=str(e)))
            ok = None if ok else ok
            continue
        ok = ok and rep['all_equal']
        n_equal = sum(x['equal'] for x in rep['models'])
        summary.append(dict(input=str(p), dataset=rep['dataset'], algo=rep['algo'], n_checked=rep['n_checked'], n_equal=n_equal,
                            all_equal=rep['all_equal'], rounds=sorted({x['rounds'] for x in rep['models']}),
                            wall_s=rep['wall_s'], out=str(out)))
        print(f"{rep['dataset']} {PROCEDURES.get(rep['algo'], rep['algo'])} {n_equal} of {rep['n_checked']} equal", flush=True)
    (out_dir / 'alone_summary.json').write_text(json.dumps(dict(k=k, all_equal=ok, results=summary), indent=1), encoding='utf-8')
    print('all equal' if ok else ('reconstruction error' if ok is None else 'mismatch'), flush=True)
    sys.exit(0 if ok else (2 if ok is None else 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('check', choices=['cpu', 'gpu', 'full', 'alone'])
    p.add_argument('results', nargs='*')
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--out-dir', default='out/alone')
    a = p.parse_args()
    if a.check == 'full':
        from scripts import check_full
        check_full.main()
    elif a.check == 'alone':
        if not a.results:
            p.error('alone needs at least one search result file')
        alone(a.results, a.k, a.out_dir)
    else:
        quick(a.check)


if __name__ == '__main__':
    main()
