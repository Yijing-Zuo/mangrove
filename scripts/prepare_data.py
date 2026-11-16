import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / 'bench' / 'prep'
sys.path.insert(0, str(PREP))
READ_BY_RUNS = ('y', 'folds', 'Xb_fold', 'nbins_fold', 'Xb_global', 'nbins_global', 'group', 'cut_ptr_fold', 'cut_val_fold')


class IntegrityError(RuntimeError):
    pass


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def year_from_uci(cache):
    import sources
    d = Path(cache) / 'uci_year'
    d.mkdir(parents=True, exist_ok=True)
    txt = d / 'YearPredictionMSD.txt'
    if not txt.exists():
        z = d / 'y.zip'
        subprocess.run(['curl', '-sS', '-L', '--retry', '8', '--retry-delay', '5', '-o', str(z),
                        'https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip'], check=True, timeout=1800)
        zipfile.ZipFile(z).extract('YearPredictionMSD.txt', d)
    target = Path(cache) / 'verified_arff' / '44027' / 'dataset.arff'
    target.parent.mkdir(parents=True, exist_ok=True)
    with txt.open() as f, target.open('w') as o:
        o.write('@relation year\n' + ''.join(f"@attribute '{i}' numeric\n" for i in range(1, 91)) +
                "@attribute 'year' numeric\n@data\n")
        for line in f:
            c = line.rstrip('\n').split(',')
            o.write(','.join(c[1:] + [c[0]]) + '\n')
    st = target.stat()
    target.with_suffix('.receipt.json').write_text(json.dumps({'url': 'uci', 'md5': sources.PINNED_ARFF[44027]['md5'],
                                                               'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}))


def higgs_source(cache, feature_names, csv_gz):
    import urllib.request
    import sources
    from higgs_uci_fallback import Higgs11MUCI
    if csv_gz:
        return Higgs11MUCI(csv_gz, feature_names)
    try:
        req = urllib.request.Request(sources.HIGGS11M['parquet_url'], method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status == 200:
                return sources.Higgs11M(cache)
    except Exception as e:
        print('parquet host:', repr(e), flush=True)
    d = Path(cache) / 'uci_higgs_280'
    d.mkdir(parents=True, exist_ok=True)
    z = d / 'higgs.zip'
    subprocess.run(['curl', '-sS', '-L', '--retry', '8', '--retry-delay', '5', '-C', '-', '-o', str(z),
                    'https://archive.ics.uci.edu/static/public/280/higgs.zip'], check=True, timeout=2400)
    zipfile.ZipFile(z).extractall(d)
    return Higgs11MUCI(d / 'HIGGS.csv.gz', feature_names)


def save_raw(path, Xnum, folds, F, cat, max_bin):
    arrays = {'Xnum': Xnum, 'folds': np.asarray(folds, dtype=np.int8), 'n_folds': np.array(F),
              'max_bin': np.array(max_bin), 'cat_columns': np.array(sorted(cat or {}), dtype=np.int64)}
    for j, (codes, labels) in (cat or {}).items():
        arrays[f'codes_{j}'] = np.asarray(codes, dtype=np.int64)
        arrays[f'labels_{j}'] = np.array([str(x) for x in labels], dtype=str)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)


def prepare_one(name, data_dir, raw_dir, cache, ref, workers):
    import binning
    import prep_all
    import fast_binning as FB
    js = data_dir / f'{name}.json'
    if not js.exists():
        js.write_text(json.dumps({'npz_compression': ref['npz_compression']}))
    orig, first, t0 = binning.fold_local, [True], time.time()
    raw = raw_dir / f'{name}.npz'

    def hook(Xnum, folds, F, cat=None, mb=binning.MAX_BIN):
        if not first[0]:
            return orig(Xnum, folds, F, cat, mb)
        first[0] = False
        save_raw(raw, Xnum, folds, F, cat, mb)
        return FB.fold_local_mask_sortonce(Xnum, [folds != k for k in range(F)], cat, mb, workers=workers)

    binning.fold_local = hook
    try:
        prep_all.run_one(name, data_dir, cache)
    finally:
        binning.fold_local = orig
    if first[0]:
        raise RuntimeError('the preparation never binned fold-locally')
    npz = data_dir / f'{name}.npz'
    got = sha256(npz)
    other = []
    if got != ref['file_sha256']:
        with np.load(npz, allow_pickle=False) as z:
            have = {k: hashlib.sha256(np.ascontiguousarray(z[k]).data).hexdigest() for k in z.files}
        want = ref['arrays']
        differ = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
        fatal = [k for k in differ if k in READ_BY_RUNS]
        if fatal:
            raise IntegrityError(f'{name}: arrays differ from the paper runs: {fatal}')
        other = differ
    with np.load(raw, allow_pickle=False) as rz, np.load(npz, allow_pickle=False) as z:
        Xnum, folds, F, mb = rz['Xnum'], rz['folds'], int(rz['n_folds']), int(rz['max_bin'])
        cat = {int(j): (rz[f'codes_{j}'], rz[f'labels_{j}']) for j in rz['cat_columns']}
        Xbf, nbf, _, _ = FB.fold_local_mask_sortonce(Xnum, [folds != k for k in range(F)], cat, mb, workers=workers)
        if not (np.array_equal(Xbf, z['Xb_fold']) and np.array_equal(nbf, z['nbins_fold'])):
            raise IntegrityError(f'{name}: bins from the raw file differ from the prepared file')
    return {'table': name, 'ok': True, 'sha256': got, 'identical_file': got == ref['file_sha256'],
            'other_arrays_differ': other, 'seconds': round(time.time() - t0, 1)}


def main():
    refs = json.loads((PREP / 'digests.json').read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument('tables', nargs='*')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    ap.add_argument('--cache', type=Path, default=ROOT / 'data' / 'cache')
    ap.add_argument('--higgs-csv-gz', type=Path)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    names = a.tables or list(refs)
    raw_dir = a.data_dir / 'raw'
    for d in (a.data_dir, raw_dir, a.cache):
        d.mkdir(parents=True, exist_ok=True)
    if 'year' in names:
        year_from_uci(a.cache)
    big = [n for n in names if n in ('higgs-2m', 'higgs-4m', 'higgs-11m')]
    if big:
        import sources
        H = higgs_source(a.cache, refs[big[0]]['feature_names'], a.higgs_csv_gz)
        sources.Higgs11M = lambda cache: H
    report = []
    for name in names:
        try:
            r = prepare_one(name, a.data_dir, raw_dir, a.cache, refs[name], a.workers)
        except Exception:
            r = {'table': name, 'ok': False, 'error': traceback.format_exc()}
            for p in (raw_dir / f'{name}.npz', a.data_dir / f'{name}.npz', a.data_dir / f'{name}.json'):
                p.unlink(missing_ok=True)
        print(json.dumps(r), flush=True)
        report.append(r)
    (a.data_dir / 'prepare_report.json').write_text(json.dumps(report, indent=1))
    print('prepared', sum(r['ok'] for r in report), 'of', len(report), 'tables')


if __name__ == '__main__':
    main()
