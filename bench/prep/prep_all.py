import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import io_utils as U
import prep_main, prep_large, prep_breadth, prep_grouped, synth

REGISTRY = {}
for s in prep_main.MAIN_SPECS:
    REGISTRY[s[0]] = dict(group="main", fn=prep_main.prepare, source_id=s[1],
                          experiments=["main"] + (["grouped"] if s[4] else []))
for k, v in prep_large.LARGE_SPECS.items():
    REGISTRY[k] = dict(group="scaling", fn=prep_large.prepare, source_id=v["did"], optional=v.get("optional", False),
                       experiments=["scaling"] + (["grouped"] if v.get("group") else []))
for k, v in prep_breadth.BREADTH_SPECS.items():
    REGISTRY[k] = dict(group="breadth", fn=prep_breadth.prepare, source_id=v["openml_id"], experiments=["breadth"])
for k, v in prep_grouped.GROUPED_SPECS.items():
    REGISTRY[k] = dict(group="grouped", fn=prep_grouped.prepare, source_id=v["did"], experiments=["grouped"])
for k, v in synth.INSTANCES.items():
    REGISTRY[k] = dict(group="synthetic", fn=synth.prepare, source_id="prep/synth.py",
                       experiments=["synthetic"] + (["grouped"] if v.get("n_groups") else []))


def run_one(nm, out, cache, n_folds=None):
    r = REGISTRY[nm]
    base, nf = (r["base"], r["n_folds"]) if "base" in r else (nm, n_folds)
    if not nf or nf == U.N_FOLDS:
        return REGISTRY[base]["fn"](base, out, cache)
    U.VARIANT.update(n_folds=nf, suffix=f"-f{nf}")
    try:
        return REGISTRY[base]["fn"](base, out, cache)
    finally:
        U.VARIANT.update(n_folds=None, suffix="")
