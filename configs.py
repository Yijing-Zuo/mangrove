MAIN = ['adult', 'higgs-50k', 'road-safety', 'medical-charges', 'sgemm', 'particulate-matter', 'year', 'airlines',
        'covertype', 'higgs-1m']
SMALL = ['adult', 'higgs-50k']
EXTRA_TABLES = ['miniboone', 'jannis', 'diamonds', 'jungle-chess', 'covertype-binary']
DEPTH_TABLES = ['higgs-1m', 'year', 'sgemm']
POOL_TABLES = ['adult', 'higgs-50k', 'sgemm', 'higgs-1m']
LARGE = ['higgs-1m', 'higgs-2m', 'higgs-4m', 'higgs-11m']
ROWS = ['higgs-50k', 'higgs-100k', 'higgs-500k', 'higgs-1m']
GROUPED = ['house-sales-zipcode', 'airlines-airport', 'airlines', 'adult', 'bank-marketing', 'covertype',
           'synth-grouped-50k']
PROCEDURES = ['random', 'sha', 'asha', 'hyperband-parallel']
SEEDS = [0, 1, 2]
SEARCH = {'n_configs': 128, 'r_min': 3, 'r_max': 81, 'eta': 3, 'cuts': 'fold', 'depth': 'fixed6',
          'folds': '0,1,2,3,4', 'gpu_sample_ms': 1000}
SAMPLING = {'subsample': '0.5,1', 'colsample': '0.5,1'}
MANGROVE = {'slots': 320, 'device': 'cuda', 'backend': 'gpu', 'memory_budget_gb': 16, 'alloc_cap_factor': 1.5}
FIXED = {'rounds': 100, 'folds': '0,1,2,3,4', 'gpu_sample_ms': 1000}
FIXED_CONFIGS = {'adult': 64, 'california': 64, 'higgs-50k': 64, 'sgemm': 32, 'higgs-1m': 16, 'year': 16}
BUDGETS = {('covertype', 'sha'): [3, 5, 10], ('higgs-1m', 'sha'): [5, 10], ('higgs-1m', 'hyperband-parallel'): [5],
           ('year', 'sha'): [5], ('year', 'hyperband-parallel'): [5, 10], ('sgemm', 'sha'): [5],
           ('sgemm', 'hyperband-parallel'): [5]}
SMALL_BUDGETS = {'adult': [1, 1.5], 'higgs-50k': [1.5, 2.5]}
SETTINGS = [{'n_configs': 64}, {'n_configs': 256}, {'eta': 2}, {'eta': 4}, {'r_max': 27}, {'r_max': 243}]
ARMS = {'full': ({}, {}), 'at-most-80': ({'slots': 80}, {}), 'at-most-20': ({'slots': 20}, {}),
        'no-node-maps': ({}, {'MANGROVE_NODEMAP': '0', 'MANGROVE_NODEMAP_REQUIRE': None}),
        'no-fused-sampling': ({}, {'MANGROVE_FUSED_SAMPLING': '0'}),
        'no-pinned-reuse': ({}, {'MANGROVE_PINNED_LIFECYCLE': '0'}),
        'no-fused-gradients': ({}, {'MANGROVE_FUSED_GRADIENTS': '0'})}
POOLS = [(1, 1, True), (2, 1, True), (4, 1, True), (8, 1, True), (16, 1, True), (4, 3, True), (1, 12, False)]
MODEL_COUNTS = [1, 2, 4, 8, 16, 32, 64]

ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
ENV_MANGROVE = {'MANGROVE_NODEMAP_REQUIRE': '1', 'MANGROVE_BENCH_INPROC_BIN': 'data/raw'}
ENV_MANGROVE_SEARCH = {**ENV_MANGROVE, 'MANGROVE_BENCH_MG_FASTSCORE': '1', 'MANGROVE_BENCH_MG_LIFEREC': '1'}
ENV_XGB_SEARCH = {'MANGROVE_BENCH_XGB_FASTSCORE': '1', 'MANGROVE_BENCH_XGB_WARMUP': 'all'}


def flags(d):
    out = []
    for k, v in d.items():
        if v is True:
            out.append('--' + k.replace('_', '-'))
        elif v is not None and v is not False:
            out += ['--' + k.replace('_', '-'), str(v)]
    return out


def search(study, table, proc, seed, system, out, sampling=False, extra=None, env=None, **over):
    s = {**SEARCH, **over}
    argv = ['--algo', proc, '--dataset', table, '--config-seed', str(seed), '--sched-seed', str(seed),
            '--rep', str(seed), '--tag', study, '--out', out] + flags(s)
    if proc == 'asha':
        argv += ['--variant', 'promote']
    if system == 'mangrove':
        m = {**MANGROVE, **(extra or {})}
        argv = ['--trainer', 'mangrove'] + argv + flags({k: m[k] for k in ('slots', 'device', 'backend')}) + \
            ['--no-fallback', '--memory-budget-gb', str(m['memory_budget_gb']), '--no-phases',
             '--alloc-cap-factor', str(m['alloc_cap_factor'])]
        e, mps = {**ENV_MANGROVE_SEARCH, **(env or {})}, False
    else:
        p, t = (extra or {}).get('P', 8), (extra or {}).get('nthread', 1)
        argv = ['--trainer', system] + argv + ['--P', str(p), '--nthread', str(t)]
        if system == 'catboost_gpu':
            argv += ['--cb-regression-boost-from-average']
            e = {'MANGROVE_BENCH_XGB_FASTSCORE': '1'}
        elif system in ('xgb_gpu', 'xgb_gpu_tuned'):
            e = dict(ENV_XGB_SEARCH)
        else:
            e = {}
        e.update(env or {})
        mps = system in ('xgb_gpu', 'xgb_gpu_tuned') and not (p == 1 and t > 1)
    if sampling:
        argv += flags(SAMPLING)
    return {'script': 'search', 'argv': argv, 'env': e, 'mps': mps, 'out': out}


def fixed(study, table, system, out, n_configs, rep, P=None, nthread=None, mps=None, env=None, **over):
    argv = ['--method', system, '--dataset', table, '--n-configs', str(n_configs), '--tag', study, '--rep', str(rep),
            '--out', out] + flags({**FIXED, **over})
    if system == 'mangrove':
        argv += ['--no-fallback', '--memory-budget-gb', '16', '--no-phases', '--alloc-cap-factor', '1.1']
        e, m = {**ENV_MANGROVE, **(env or {})}, False
    else:
        if P is not None:
            argv += ['--P', str(P), '--nthread', str(nthread or 1)]
        e = dict(env or {})
        m = mps if mps is not None else (system == 'xgb_gpu')
    return {'script': 'fixed', 'argv': argv, 'env': e, 'mps': m, 'out': out}


def _searches(study, tables, procs, seeds, systems, sampling=False, **kw):
    runs = []
    for seed in seeds:
        for table in tables:
            for proc in procs:
                for system, extra in systems:
                    tag = system if not extra else f"{system}_{extra.get('P', '')}x{extra.get('nthread', '')}"
                    out = f'out/{study}/{table}/{proc}_{tag}_s{seed}.json'
                    runs.append(search(study, table, proc, seed, system, out, sampling, extra, **kw))
    return runs


def breadth_tables():
    import json
    from pathlib import Path
    sel = json.loads((Path(__file__).parent / 'bench' / 'prep' / 'breadth_selection.json').read_text())
    return [s['name'] for s in sel['selected']]


def study(name):
    mg, xgb, tuned = ('mangrove', None), ('xgb_gpu', {'P': 8, 'nthread': 1}), ('xgb_gpu_tuned', {'P': 8, 'nthread': 1})
    if name == 'sampling':
        runs = _searches(name, MAIN, PROCEDURES, SEEDS, [mg, xgb], sampling=True)
        runs += _searches(name, SMALL, ['hyperband-sequential'], SEEDS, [mg, xgb], sampling=True)
        for slots in (80, 1024):
            for t in SMALL:
                for p in ('sha', 'hyperband-parallel'):
                    out = f'out/{name}/{t}/{p}_mangrove_at{slots}_s0.json'
                    runs.append(search(name, t, p, 0, 'mangrove', out, True, {'slots': slots}))
        return runs
    if name == 'no-sampling':
        runs = _searches(name, MAIN, PROCEDURES, SEEDS, [mg, xgb])
        return runs + _searches(name, SMALL, ['hyperband-sequential'], SEEDS, [mg, xgb])
    if name == 'tuned':
        # the 42 sampled workloads of `sampling`; the tuned pool builds its matrices on the GPU where they have the
        # host-built cuts (SGEMM, Year, Higgs-1M) and on the host elsewhere
        env = lambda t: None if t in ('sgemm', 'year', 'higgs-1m') else {'MANGROVE_BENCH_XGBT_DEVICE_DATA': '0'}
        runs = []
        for tables, procs in ((MAIN, PROCEDURES), (SMALL, ['hyperband-sequential'])):
            for seed in SEEDS:
                for t in tables:
                    for p in procs:
                        runs += _searches(name, [t], [p], [seed], [mg, xgb], sampling=True)
                        runs += _searches(name, [t], [p], [seed], [tuned], sampling=True, env=env(t))
        return runs
    if name == 'pool16':
        return _searches(name, ['higgs-1m', 'sgemm'], PROCEDURES, [0], [('xgb_gpu', {'P': 16, 'nthread': 1})])
    if name == 'extra-tables':
        return _searches(name, EXTRA_TABLES, PROCEDURES, SEEDS, [mg, xgb])
    if name == 'depths':
        return _searches(name, DEPTH_TABLES, PROCEDURES, SEEDS, [mg, xgb], depth='mixed3-7')
    if name == 'settings':
        runs = []
        for s in SETTINGS:
            k, v = next(iter(s.items()))
            for t in ('higgs-1m', 'sgemm'):
                for p in ('sha', 'hyperband-parallel'):
                    for system, extra in (mg, xgb):
                        out = f'out/{name}/{t}/{p}_{k}{v}_{system}_s0.json'
                        runs.append(search(name, t, p, 0, system, out, False, extra, **s))
        return runs
    if name == 'lightgbm-catboost':
        runs = _searches(name, ['sgemm', 'medical-charges', 'year'], PROCEDURES, [0],
                         [mg, ('lgb_cpu', {'P': 12, 'nthread': 1})])
        runs += _searches(name, ['sgemm', 'medical-charges', 'year'], ['random'], [0],
                          [('catboost_gpu', {'P': 1, 'nthread': 1})])
        for t in ('adult', 'california', 'higgs-50k', 'sgemm', 'higgs-1m', 'year'):
            runs.append(fixed(name, t, 'lgb_cpu', f'out/{name}/{t}/fixed_lgb_cpu.json', FIXED_CONFIGS[t], 0, 12, 1))
        for t in ('adult', 'sgemm', 'higgs-1m'):
            runs.append(fixed(name, t, 'catboost_gpu', f'out/{name}/{t}/fixed_catboost_gpu.json', FIXED_CONFIGS[t], 0,
                              mps=False))
        for t in ('adult', 'california', 'higgs-50k', 'sgemm', 'higgs-1m', 'year'):
            for r in SEEDS:
                runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/fixed_mangrove_r{r}.json', FIXED_CONFIGS[t], r))
        return runs
    if name == 'cpu':
        runs = []
        for t in ('adult', 'california', 'higgs-50k'):
            for lib in ('xgb_cpu', 'lgb_cpu'):
                for p, n in ((12, 1), (6, 2), (3, 4), (1, 12)):
                    runs.append(fixed(name, t, lib, f'out/{name}/{t}/{lib}_{p}x{n}.json', 64, 0, p, n, False))
            for r in SEEDS:
                runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_r{r}.json', 64, r))
        return runs
    if name == 'openboost':
        runs = []
        for t, n, s, box in (('california', 64, 4, 2400), ('sgemm', 32, 4, 2400), ('year', 16, 8, 3900)):
            runs.append(fixed(name, t, 'openboost_gpu', f'out/{name}/{t}/openboost.json', n, 0, mps=False, folds='0',
                              stride=s, group_size=32, memory_budget_gb=16, time_box_s=box))
            for r in SEEDS:
                runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_r{r}.json', n, r, folds='0', stride=s))
        return runs
    if name == 'breadth':
        runs = []
        for t in breadth_tables():
            runs.append(fixed(name, t, 'mangrove', f'out/{name}/warm/{t}.json', 32, 0))
            for r in SEEDS:
                runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_r{r}.json', 32, r))
                runs.append(fixed(name, t, 'xgb_gpu', f'out/{name}/{t}/xgb_gpu_8_r{r}.json', 32, r, 8, 1))
            runs.append(fixed(name, t, 'xgb_gpu', f'out/{name}/{t}/xgb_gpu_16_r0.json', 32, 0, 16, 1))
        return runs
    if name == 'pools':
        runs = []
        for r in SEEDS:
            for t in POOL_TABLES:
                for p, n, m in sorted(POOLS, key=lambda x: x[2]):
                    runs.append(fixed(name, t, 'xgb_gpu', f'out/{name}/{t}/xgb_gpu_{p}x{n}_r{r}.json', 64, r, p, n, m))
                runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_r{r}.json', 64, r))
        return runs
    if name == 'groups':
        runs = []
        for t in GROUPED:
            g = {'kind': 'grouped', 'group_val_fold': 0, 'min_group_rows': 20, 'folds': None}
            if t == 'airlines-airport':
                g.update(min_group_size=100, stride=4)
            for r in SEEDS:
                runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_r{r}.json', 16, r, **g))
                runs.append(fixed(name, t, 'xgb_cpu', f'out/{name}/{t}/xgb_cpu_12x1_r{r}.json', 16, r, 12, 1, False, **g))
                runs.append(fixed(name, t, 'xgb_gpu', f'out/{name}/{t}/xgb_gpu_16_r{r}.json', 16, r, 16, 1, **g))
                runs.append(fixed(name, t, 'xgb_gpu', f'out/{name}/{t}/xgb_gpu_8_r{r}.json', 16, r, 8, 1, **g))
        return runs
    if name == 'models':
        runs = []
        for t in SMALL:
            for r in SEEDS:
                for m in MODEL_COUNTS:
                    runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_M{m}_r{r}.json', m, r, folds='0'))
                for c in (32, 64, 320):
                    runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_M{5 * c}_r{r}.json', c, r))
                    for p in (8, 16):
                        runs.append(fixed(name, t, 'xgb_gpu', f'out/{name}/{t}/xgb_gpu_{p}_M{5 * c}_r{r}.json', c, r, p, 1))
        return runs
    if name == 'rows':
        runs = []
        for t in ROWS:
            for r in SEEDS:
                runs.append(fixed(name, t, 'mangrove', f'out/{name}/{t}/mangrove_r{r}.json', 64, r))
                for p in (8, 16):
                    runs.append(fixed(name, t, 'xgb_gpu', f'out/{name}/{t}/xgb_gpu_{p}_r{r}.json', 64, r, p, 1))
        return runs
    if name == 'large':
        runs = []
        env = {'MANGROVE_BENCH_MEM_FRACTION': '0.92'}
        for t in LARGE:
            for p in ('sha', 'hyperband-parallel'):
                runs.append(search(name, t, p, 0, 'mangrove', f'out/{name}/{t}/{p}_mangrove.json', False,
                                   {'memory_budget_gb': 60}, env=env))
                for P, n in ((8, 1), (4, 3), (1, 12)):
                    runs.append(search(name, t, p, 0, 'xgb_gpu', f'out/{name}/{t}/{p}_xgb_gpu_{P}x{n}.json', False,
                                       {'P': P, 'nthread': n}, env=env))
        return runs
    if name == 'budgets':
        runs = []
        for (t, p), gbs in BUDGETS.items():
            for gb in gbs:
                runs.append(search(name, t, p, 0, 'mangrove', f'out/{name}/{t}/{p}_mangrove_{gb}g.json', False,
                                   {'memory_budget_gb': gb}, env={'MANGROVE_BENCH_INPROC_BIN': None}))
        for t, gbs in SMALL_BUDGETS.items():
            for p in ('sha', 'hyperband-parallel'):
                for gb in gbs:
                    runs.append(search(name, t, p, 0, 'mangrove', f'out/{name}/{t}/{p}_mangrove_{gb}g_sampled.json',
                                       True, {'memory_budget_gb': gb, 'alloc_cap_factor': 3}))
        return runs
    if name in ('ablation', 'ablation-no-sampling'):
        sampled = name == 'ablation'
        arms = {k: v for k, v in ARMS.items() if sampled or k in ('full', 'no-node-maps')}
        runs = []
        for t in POOL_TABLES:
            runs.append(search(name, t, 'sha', 0, 'mangrove', f'out/{name}/{t}/warm.json', sampled))
            for seed in SEEDS:
                for p in ('sha', 'hyperband-parallel'):
                    for arm, (extra, env) in arms.items():
                        runs.append(search(name, t, p, seed, 'mangrove', f'out/{name}/{t}/{p}_{arm}_s{seed}.json',
                                           sampled, extra, env=env))
        return runs
    if name == 'trace':
        out = 'out/trace/higgs-1m/sha_mangrove'
        return [search(name, 'higgs-1m', 'sha', 0, 'mangrove', out + '_warm.json', True),
                search(name, 'higgs-1m', 'sha', 0, 'mangrove', out + '_traced.json', True,
                       env={'MANGROVE_BENCH_MG_STEPTRACE': '1'}),
                search(name, 'higgs-1m', 'sha', 0, 'mangrove', out + '_plain.json', True)]
    if name == 'anatomy':
        return [{'script': 'anatomy', 'argv': ['--data', t, '--configs', str(c), '--folds', f, '--nodemap', '1',
                                               '--out', f'out/anatomy/{t}_M{m}.json'], 'env': {}, 'mps': False,
                 'out': f'out/anatomy/{t}_M{m}.json'}
                for t, c, f, m in (('higgs-1m', 1, '0', 1), ('higgs-1m', 16, '0,1,2,3,4', 80),
                                   ('adult', 64, '0,1,2,3,4', 320))]
    raise KeyError(name)


STUDIES = ['sampling', 'no-sampling', 'tuned', 'pool16', 'extra-tables', 'depths', 'settings', 'lightgbm-catboost',
           'cpu', 'openboost', 'breadth', 'pools', 'groups', 'models', 'rows', 'large', 'budgets', 'ablation',
           'ablation-no-sampling', 'trace', 'anatomy']
