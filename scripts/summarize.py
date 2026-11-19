import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROC = {'grid': 'random', 'sha': 'sha', 'asha': 'asha', 'hbpar': 'hyperband-parallel', 'hyperband': 'hyperband-sequential'}


def gmean(v):
    v = list(v)
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float('nan')


def load(root):
    root = Path(root)
    wall = {}
    for t in root.rglob('timing.jsonl'):
        for line in t.read_text().splitlines():
            r = json.loads(line)
            if r['rc'] == 0:
                wall[Path(r['out']).name, Path(r['out']).parent.name] = r['elapsed_s']
    runs = []
    for p in root.rglob('*.json'):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(r, dict) or r.get('status') != 'ok' or 'timing' not in r:
            continue
        r['_wall'] = wall.get((p.name, p.parent.name))
        r['_path'] = p
        runs.append(r)
    return runs


def system(r):
    v = r.get('variant') or {}
    t = v.get('trainer') or r.get('method')
    return t if t == 'mangrove' else f"{t}_{v.get('P')}x{v.get('nthread')}"


def searches(root, base='xgb_gpu_8x1', target='mangrove'):
    cells = defaultdict(lambda: defaultdict(list))
    for r in load(root):
        if 'dynamic_summary' not in r:
            continue
        v = r['variant']
        if v.get('trainer') == 'mangrove' and v.get('slots') != 320:
            continue
        key = (r['workload']['dataset'], PROC.get(v['algo'], v['algo']))
        cells[key][system(r)].append(r)
    rows = []
    for key, by in sorted(cells.items()):
        px = {r['repetition']: r for r in by.get(base, [])}
        pairs = [(a, px[a['repetition']]) for a in sorted(by.get(target, []), key=lambda r: r['repetition']) if a['repetition'] in px]
        if not pairs:
            continue
        mg, xg = [a for a, _ in pairs], [b for _, b in pairs]
        med = lambda rs, f: statistics.median(f(r) for r in rs)
        row = {'table': key[0], 'procedure': key[1], 'seeds': len(pairs),
               'train': med(xg, lambda r: r['timing']['train_s']) / med(mg, lambda r: r['timing']['train_s']),
               'work': statistics.median(b['throughput']['model_rounds_executed'] / a['throughput']['model_rounds_executed'] for a, b in pairs)}
        if all(r['_wall'] for r in mg + xg):
            row['end_to_end'] = med(xg, lambda r: r['_wall']) / med(mg, lambda r: r['_wall'])
        q = []
        for a, b in pairs:
            x, y = a['dynamic_summary']['final_best_primary'], b['dynamic_summary']['final_best_primary']
            if x is not None and y:
                q.append(abs(x - y) / abs(y))
        row['quality'] = q
        row['resident'] = statistics.mean(r['dynamic_summary']['trainer'].get('mean_active_models') or 0 for r in mg)
        rows.append(row)
    for row in rows:
        print(f"{row['table']:20s} {row['procedure']:22s} end-to-end {row.get('end_to_end', float('nan')):.3f}x  "
              f"loop {row['train']:.3f}x  equal work {row.get('end_to_end', float('nan')) / row['work']:.3f}x  "
              f"resident {row['resident']:.1f}")
    groups = [('all', rows)] + [(p, [r for r in rows if r['procedure'] == p]) for p in PROC.values()]
    for name, rs in groups:
        rs = [r for r in rs if 'end_to_end' in r]
        if not rs:
            continue
        e = [r['end_to_end'] for r in rs]
        print(f"{name:22s} {len(rs)} workloads  end-to-end {gmean(e):.2f}x ({min(e):.2f}-{max(e):.2f})  "
              f"loop {gmean(r['train'] for r in rs):.2f}x  equal work {gmean(r['end_to_end'] / r['work'] for r in rs):.2f}x  "
              f"faster {sum(x > 1 for x in e)}/{len(e)}")
    q = [x for r in rows for x in r['quality']]
    if q:
        print(f'best score difference over {len(q)} searches: max {100 * max(q):.3f}%  median {100 * statistics.median(q):.3f}%')


def breakdown(root, algo='sha', base='xgb_gpu_8x1'):
    cells = defaultdict(lambda: defaultdict(list))
    for r in load(root):
        v = r['variant']
        if 'dynamic_summary' not in r or v.get('algo') != algo or not r['_wall']:
            continue
        if v.get('trainer') == 'mangrove' and v.get('slots') != 320:
            continue
        cells[r['workload']['dataset']][system(r)].append(r)
    for t, by in sorted(cells.items()):
        line = [f'{t:20s}']
        for s in (base, 'mangrove'):
            rs = by.get(s)
            if not rs:
                continue
            w = statistics.median(r['_wall'] for r in rs)
            p = statistics.median(r['timing']['prep_s'] for r in rs)
            tr = statistics.median(r['timing']['train_s'] for r in rs)
            line.append(f'{s} preparation {p:.1f} loop {tr:.1f} other {w - p - tr:.1f} end-to-end {w:.1f} s')
        print('  '.join(line))


def fixed(root, base=None):
    cells = defaultdict(lambda: defaultdict(list))
    for r in load(root):
        if 'dynamic_summary' in r:
            continue
        cells[r['workload']['dataset']][system(r)].append(r)
    out = defaultdict(list)
    for t, by in sorted(cells.items()):
        if 'mangrove' not in by:
            continue
        mw = statistics.median(r['_wall'] for r in by['mangrove'] if r['_wall'])
        mt = statistics.median(r['timing']['train_s'] for r in by['mangrove'])
        line = [t]
        for s, rs in sorted(by.items()):
            if s == 'mangrove' or (base and not s.startswith(base)):
                continue
            w = [r['_wall'] for r in rs if r['_wall']]
            e = statistics.median(w) / mw if w else float('nan')
            tr = statistics.median(r['timing']['train_s'] for r in rs) / mt
            out[s].append((e, tr))
            line.append(f'{s} end-to-end {e:.2f}x loop {tr:.2f}x')
        print('  '.join(line))
    for s, v in sorted(out.items()):
        e = [x[0] for x in v]
        print(f'{s}: {len(v)} tables  end-to-end {gmean(e):.2f}x ({min(e):.2f}-{max(e):.2f})  '
              f'loop {gmean(x[1] for x in v):.2f}x')


def pools(root):
    cells = defaultdict(lambda: defaultdict(list))
    for r in load(root):
        cells[r['workload']['dataset']][system(r)].append(r)
    for t, by in sorted(cells.items()):
        n = by['mangrove'][0]['n_models'] if 'mangrove' in by else None
        ref_w = statistics.median(r['_wall'] for r in by['xgb_gpu_1x1'])
        ref_t = statistics.median(r['timing']['train_s'] for r in by['xgb_gpu_1x1'])
        for s, rs in sorted(by.items()):
            w = statistics.median(r['_wall'] for r in rs)
            tr = statistics.median(r['timing']['train_s'] for r in rs)
            p = statistics.median(r['timing']['prep_s'] for r in rs)
            print(f'{t:12s} {s:14s} end-to-end {ref_w / w:.2f}x  loop {ref_t / tr:.2f}x  preparation {p:.1f} s  models {n}')


def models(root):
    cells = defaultdict(list)
    for r in load(root):
        if system(r) == 'mangrove':
            cells[r['workload']['dataset'], r['n_models']].append(r['timing']['train_s'])
    for t in sorted({k[0] for k in cells}):
        T = {m: statistics.median(v) / 100 for (tt, m), v in cells.items() if tt == t}
        for m in sorted(T):
            print(f'{t:12s} {m:5d} models  {1e3 * T[m]:.2f} ms per round')
        if 160 in T and 1600 in T:
            c = (T[1600] - T[160]) / 1440
            print(f'{t}: F = {1e3 * (T[160] - 160 * c):.2f} ms, c = {1e3 * c:.5f} ms per model')
    fixed(root)


def arms(root):
    cells = defaultdict(lambda: defaultdict(list))
    for r in load(root):
        name = r['_path'].stem.rsplit('_s', 1)[0]
        if name == 'warm':
            continue
        proc, arm = name.split('_', 1)
        cells[r['workload']['dataset'], proc][arm].append(r)
    ratio = defaultdict(list)
    same = defaultdict(lambda: [0, 0])
    for key, by in sorted(cells.items()):
        full = by.get('full')
        if not full:
            continue
        ft = statistics.median(r['timing']['train_s'] for r in full)
        ref = {(r['repetition'], m['cfg'], m['part']): m.get('tree_sha') or m.get('margin_sha') for r in full for m in r['per_model']}
        for arm, rs in by.items():
            ratio[arm].append(statistics.median(r['timing']['train_s'] for r in rs) / ft)
            for r in rs:
                for m in r['per_model']:
                    k = (r['repetition'], m['cfg'], m['part'])
                    same[arm][0] += ref.get(k) == (m.get('tree_sha') or m.get('margin_sha'))
                    same[arm][1] += 1
    for arm, v in sorted(ratio.items()):
        print(f'{arm:20s} {len(v)} workloads  loop time {gmean(v):.2f}x of full  identical models {same[arm][0]}/{same[arm][1]}')


def memory(root):
    for r in sorted(load(root), key=lambda r: str(r['_path'])):
        if 'dynamic_summary' not in r:
            continue
        tr = r['dynamic_summary']['trainer']
        ev = (tr.get('lifecycle') or {}).get('events') or {}
        print(f"{r['_path'].parent.name:12s} {r['_path'].stem:40s} budget {r['variant'].get('memory_budget_gb')} GiB  "
              f"peak {(r['memory'].get('gpu_peak_allocated_bytes') or 0) / 2 ** 30:.2f} GiB  loop {r['timing']['train_s']:.1f} s  "
              f"resident {tr.get('mean_active_models')}  lifecycle {json.dumps(ev)}")
        for s in tr.get('step_trace') or []:
            print('   ', json.dumps(s))


def anatomy(root):
    for p in sorted(Path(root).rglob('*.json')):
        r = json.loads(p.read_text())
        print(p.stem, {k: r.get(k) for k in ('wall_ms', 'wall_noprof_ms', 'gpu_kernel_ms', 'kernels')})


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('searches', 'breakdown', 'fixed', 'pools', 'models', 'ablation', 'memory', 'anatomy'):
        sys.exit('usage: python -m scripts.summarize {searches,breakdown,fixed,pools,models,ablation,memory,anatomy} <dir> [base]')
    if not Path(sys.argv[2]).is_dir():
        sys.exit(f'{sys.argv[2]}: no such directory')
    {'searches': searches, 'breakdown': breakdown, 'fixed': fixed, 'pools': pools, 'models': models, 'ablation': arms, 'memory': memory,
     'anatomy': anatomy}[sys.argv[1]](*sys.argv[2:])
