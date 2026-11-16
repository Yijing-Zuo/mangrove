import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import configs

ROOT = Path(__file__).resolve().parent.parent
CTL = 'nvidia-cuda-mps-control'


def sh(argv, inp=None, timeout=30):
    try:
        p = subprocess.run(argv, input=inp, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except Exception as e:
        return None, repr(e)


def mps_running():
    rc, out = sh(['pgrep', '-f', 'nvidia-cuda-mps'], timeout=10)
    return rc == 0 and bool(out.strip())


def mps(on):
    ctl = shutil.which(CTL)
    if on:
        if not mps_running():
            if not ctl or sh([ctl, '-d'])[0] != 0:
                raise RuntimeError('cannot start the CUDA MPS daemon')
            time.sleep(2)
        return
    if ctl and mps_running():
        sh([ctl], inp='quit\n')
        time.sleep(2)
        for _ in range(16):
            if not mps_running():
                break
            time.sleep(0.5)
    if mps_running():
        raise RuntimeError('the CUDA MPS daemon is still running')


def execute(run, log_path):
    module = {'search': 'scripts.search', 'fixed': 'scripts.fixed', 'anatomy': 'scripts.anatomy'}[run['script']]
    argv = [sys.executable, '-m', module] + run['argv']
    env = dict(os.environ, **configs.ENV)
    for k, v in run['env'].items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(ROOT / v) if k == 'MANGROVE_BENCH_INPROC_BIN' else v
    env['MANGROVE_BENCH_MPS'] = '1' if run['mps'] else '0'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, 'a', encoding='utf-8', errors='replace') as log:
        log.write(f"$ {' '.join(argv)}\n")
        log.flush()
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rc = p.wait()
    return rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('study', choices=configs.STUDIES)
    ap.add_argument('--tables')
    ap.add_argument('--only')
    ap.add_argument('--print', action='store_true')
    a = ap.parse_args()
    runs = configs.study(a.study)
    if a.tables:
        keep = set(a.tables.split(','))
        runs = [r for r in runs if r['argv'][r['argv'].index('--dataset' if '--dataset' in r['argv'] else '--data') + 1] in keep]
    if a.only:
        runs = [r for r in runs if a.only in r['out']]
    if a.print:
        for r in runs:
            env = ' '.join(f'{k}={v}' for k, v in {**r['env'], 'MPS': int(r['mps'])}.items() if v is not None)
            print(f"{env} python -m scripts.{r['script']} {' '.join(r['argv'])}")
        return
    timing = ROOT / 'out' / a.study / 'timing.jsonl'
    timing.parent.mkdir(parents=True, exist_ok=True)
    for r in runs:
        out = ROOT / r['out']
        if out.exists():
            continue
        mps(r['mps'])
        rc, elapsed = execute(r, out.with_suffix('.log'))
        rec = {'out': r['out'], 'argv': r['argv'], 'mps': r['mps'], 'rc': rc, 'elapsed_s': round(elapsed, 1)}
        with open(timing, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        print(r['out'], rc, rec['elapsed_s'], flush=True)
    mps(False)


if __name__ == '__main__':
    main()
