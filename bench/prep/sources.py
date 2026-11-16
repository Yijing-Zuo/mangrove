from pathlib import Path
import warnings
import numpy as np
import pandas as pd

PINNED_ARFF = {
    42769: {'name': 'Higgs', 'version': 3, 'file_id': '22044863', 'target': 'target',
            'url': 'https://www.openml.org/data/v1/download/22044863/Higgs.arff',
            'md5': '462a8d5b7502eef3ab2215fa6df6d8cc', 'rows': 1_000_000, 'features': 28},
    44027: {'name': 'year', 'version': 1, 'file_id': '22103115', 'target': 'year',
            'url': 'https://www.openml.org/data/v1/download/22103115/year.arff',
            'md5': 'cea7b6902adb4262fc8d2d94c2631acf', 'rows': 515_345, 'features': 90},
}


def verified_arff_path(did, cache):
    import hashlib, json, time, urllib.request
    spec = PINNED_ARFF[did]
    target = Path(cache) / 'verified_arff' / str(did) / 'dataset.arff'
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix('.receipt.json')
    if target.exists() and receipt.exists():
        rec = json.loads(receipt.read_text(encoding='utf-8'))
        st = target.stat()
        if rec.get('md5') == spec['md5'] and rec.get('bytes') == st.st_size and rec.get('mtime_ns') == st.st_mtime_ns:
            return target
    if target.exists():
        got = _md5(target)
        if got == spec['md5']:
            st = target.stat()
            receipt.write_text(json.dumps({'url': spec['url'], 'md5': got, 'bytes': st.st_size,
                                           'mtime_ns': st.st_mtime_ns}, indent=2), encoding='utf-8')
            return target
        raise IOError(f'{target}: source MD5 mismatch; preserve the file and use a fresh cache directory')
    part = target.with_suffix('.arff.part')
    h, size = hashlib.md5(), 0
    started = time.time()
    request = urllib.request.Request(spec['url'], headers={'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=120) as response, part.open('wb') as out:
        total = int(response.headers.get('Content-Length', 0)) or None
        while True:
            buf = response.read(4 * 1024 * 1024)
            if not buf:
                break
            out.write(buf); h.update(buf); size += len(buf)
            if size % (64 * 1024 * 1024) == 0:
                print(f'OpenML {did}: {size // 2**20} MiB downloaded', flush=True)
    if h.hexdigest() != spec['md5'] or (total is not None and size != total):
        raise IOError(f'OpenML {did}: ARFF MD5/length mismatch ({h.hexdigest()}, {size}/{total})')
    part.replace(target)
    st = target.stat()
    receipt.write_text(json.dumps({'url': spec['url'], 'md5': h.hexdigest(), 'bytes': size,
                                   'mtime_ns': st.st_mtime_ns, 'seconds': time.time() - started}, indent=2), encoding='utf-8')
    return target


def fetch_pinned_arff(did, cache, target=None):
    import re
    spec = PINNED_ARFF[did]
    path = verified_arff_path(did, cache)
    names, types = [], []
    with path.open('r', encoding='utf-8') as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'{path}: missing @data section')
            text = line.strip()
            if text.lower() == '@data':
                break
            match = re.match(r'''@attribute\s+('(?:[^']*)'|"(?:[^"]*)"|\S+)\s+(.+)$''', text, re.I)
            if match:
                names.append(match.group(1).strip('\"\''))
                types.append(match.group(2).strip())
        target = target or spec['target']
        if target not in names:
            raise ValueError(f'{path}: target {target!r} absent')
        dtypes = {}
        for name, kind in zip(names, types):
            if kind.lower() in ('real', 'numeric', 'integer'):
                dtypes[name] = np.float64
            elif name == target and kind.startswith('{') and kind.endswith('}'):
                labels = [x.strip().strip('\"\'') for x in kind[1:-1].split(',')]
                dtypes[name] = pd.CategoricalDtype(categories=labels)
            else:
                raise ValueError(f'{path}: unsupported attribute type {name}: {kind}')
        frame = pd.read_csv(stream, header=None, names=names, dtype=dtypes, float_precision='round_trip',
                            na_values=['?'], keep_default_na=False, comment='%', skipinitialspace=True)
    if frame.shape != (spec['rows'], spec['features'] + 1):
        raise ValueError(f'{path}: unexpected shape {frame.shape}')
    X, y = frame.drop(columns=[target]), frame[target]
    info = {'loader': 'pandas.read_csv round_trip (verified OpenML ARFF)', 'openml_id': did,
            'openml_name': spec['name'], 'openml_version': spec['version'], 'target': target,
            'openml_file_id': spec['file_id'], 'url': f'https://www.openml.org/d/{did}',
            'arff_url': spec['url'], 'arff_md5': spec['md5']}
    return X, y, [False] * len(X.columns), info


def fetch_sklearn(did, cache, target_column="default-target"):
    from sklearn.datasets import fetch_openml
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = fetch_openml(data_id=did, as_frame=True, data_home=str(Path(cache) / "sklearn"), parser="auto",
                          target_column=target_column)
    info = {"loader": "sklearn.fetch_openml", "openml_id": did, "openml_name": ds.details.get("name"),
            "openml_version": ds.details.get("version"), "openml_file_id": ds.details.get("file_id"),
            "url": f"https://www.openml.org/d/{did}"}
    return ds.data, ds.target, None, info


def fetch_openml_pkg(did, cache, target=None):
    import os
    os.environ["OPENML_SKIP_PARQUET"] = "true"
    import openml
    openml.config.set_root_cache_directory(str(Path(cache) / "openml"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                         download_features_meta_data=True)
        tgt = target or ds.default_target_attribute
        X, y, cat_ind, names = ds.get_data(target=tgt, dataset_format="dataframe")
    info = {"loader": "openml", "openml_id": did, "openml_name": ds.name, "openml_version": ds.version,
            "target": tgt, "openml_file_id": getattr(ds, "file_id", None), "url": f"https://www.openml.org/d/{did}"}
    return X, y, cat_ind, info


def fetch(did, cache, target=None):
    if did in PINNED_ARFF:
        return fetch_pinned_arff(did, cache, target)
    try:
        import openml
    except ImportError:
        X, y, ci, info = fetch_sklearn(did, cache, target_column=target or "default-target")
        info["note"] = "openml package not installed; used sklearn.fetch_openml fallback"
        return X, y, ci, info
    return fetch_openml_pkg(did, cache, target)


HIGGS11M = {"did": 45570, "parquet_url": "https://data.openml.org/datasets/0004/45570/dataset_45570.pq",
            "parquet_md5": "a78ded2b6362e52c824030797b5959ec", "parquet_bytes": 891780414,
            "arff_url": "https://www.openml.org/data/v1/download/22116554/Higgs.arff",
            "arff_md5": "3e1a963d1c8dd4aa370cef6dbe4d8fbc", "arff_bytes": 5567456609, "n": 11_000_000}


def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def download_resumable(url, dest, md5=None, size=None, attempts=30):
    import shutil, time, urllib.request
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0); f.truncate()
                shutil.copyfileobj(r, f, 1 << 22)
            if size is None or part.stat().st_size >= size:
                break
        except Exception as e:
            print(f"download attempt {attempt}: {e!r}; resuming", flush=True)
            time.sleep(5)
    if md5 is not None and _md5(part) != md5:
        raise IOError(f"md5 mismatch for {part} (expected {md5}); delete it and retry")
    part.replace(dest)
    return dest


class Higgs11M:
    def __init__(self, cache):
        import pyarrow.parquet as pq
        self.path = download_resumable(HIGGS11M["parquet_url"],
                                       Path(cache) / "openml_parquet" / "45570" / "dataset_45570.pq",
                                       HIGGS11M["parquet_md5"], HIGGS11M["parquet_bytes"])
        self.pf = pq.ParquetFile(str(self.path))
        names = list(self.pf.schema_arrow.names)
        self.target = "Target"
        self.feature_names = [c for c in names if c != self.target]
        self.n = self.pf.metadata.num_rows
        self.info = {"loader": "parquet (pyarrow, column-wise)", "openml_id": 45570, "openml_name": "Higgs",
                     "openml_version": 8, "parquet_url": HIGGS11M["parquet_url"],
                     "parquet_md5": HIGGS11M["parquet_md5"], "url": "https://www.openml.org/d/45570",
                     "origin": "UCI HIGGS (Baldi et al. 2014), 11,000,000 rows"}

    def column(self, name):
        a = self.pf.read(columns=[name]).column(0)
        return a.to_numpy() if a.num_chunks == 1 else np.concatenate([c.to_numpy(zero_copy_only=False)
                                                                        for c in a.chunks])

    def target_values(self):
        a = self.pf.read(columns=[self.target]).column(0).to_pandas()
        return a.to_numpy()
