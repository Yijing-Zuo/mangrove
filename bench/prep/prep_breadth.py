import json
import numpy as np

import io_utils as U
import sources

SEL = json.loads((U.PREP_DIR / "breadth_selection.json").read_text(encoding="utf-8"))
BREADTH_SPECS = {s["name"]: s for s in SEL["selected"]}


def prepare(name, out_dir, cache):
    sp = BREADTH_SPECS[name]
    X, y, cat_ind, info = sources.fetch(sp["openml_id"], cache)
    info["suite"] = sp["suite"]
    task = sp["task"]
    enc = U.encode_frame(X, cat_ind)
    yv, keep, classes = U.encode_target(y, task)
    if task == "binary":
        assert classes is not None and len(classes) == 2, (name, classes)
    idx = np.nonzero(keep)[0]
    enc = U.subset_rows(enc, idx)
    yv = yv[idx]
    enc, dropped = U.drop_constant_columns(enc)
    return U.write_dataset(out_dir, name, enc, yv, task, classes, info,
                           extra_meta={"group_set": "breadth", "n_source_rows": int(len(X)),
                                       "dropped_constant_columns": dropped,
                                       "class_order": "natural (numeric if labels parse as numbers, else str)"})
