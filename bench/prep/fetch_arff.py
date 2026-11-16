import argparse, hashlib, shutil, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("did", type=int)
    ap.add_argument("--cache", default=str(HERE.parent / "cache"))
    ap.add_argument("--from-file", default=None)
    a = ap.parse_args()
    import openml
    openml.config.set_root_cache_directory(str(Path(a.cache) / "openml"))
    d = openml.datasets.get_dataset(a.did, download_data=False, download_qualities=False,
                                    download_features_meta_data=False)
    dest_dir = Path(a.cache) / "openml" / "org" / "openml" / "www" / "datasets" / str(a.did)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "dataset.arff"
    part = dest_dir / "dataset.arff.part"
    if a.from_file:
        shutil.copyfile(a.from_file, part)
    else:
        url = d.url.replace("https://openml.org/", "https://www.openml.org/")
        for attempt in range(20):
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                    if have and r.status != 206:
                        f.seek(0); f.truncate()
                    shutil.copyfileobj(r, f, 1 << 20)
                break
            except Exception as e:
                print(f"attempt {attempt}: {e!r}; resuming", flush=True)
                time.sleep(5)
    got = md5(part)
    if got != d.md5_checksum:
        sys.exit(f"md5 mismatch: got {got}, expected {d.md5_checksum} ({part})")
    part.replace(dest)
    print(f"installed {dest} (md5 ok)")


if __name__ == "__main__":
    main()
