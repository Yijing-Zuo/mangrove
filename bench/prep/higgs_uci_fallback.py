import time

import numpy as np


class Higgs11MUCI:
    def __init__(self, csv_gz, feature_names):
        import pyarrow as pa
        import pyarrow.csv as pc
        cols = ["Target"] + list(feature_names)
        t0 = time.time()
        self.tbl = pc.read_csv(str(csv_gz), read_options=pc.ReadOptions(column_names=cols, block_size=1 << 26),
                               convert_options=pc.ConvertOptions(column_types={c: pa.float64() for c in cols}))
        self.target = "Target"
        self.feature_names = list(feature_names)
        self.n = self.tbl.num_rows
        self.read_s = round(time.time() - t0, 1)
        self.info = {"loader": "UCI HIGGS.csv.gz (pyarrow, float64); fallback while the OpenML parquet host answered 503",
                     "openml_id": 45570, "url": "https://www.openml.org/d/45570",
                     "uci_url": "https://archive.ics.uci.edu/static/public/280/higgs.zip",
                     "origin": "UCI HIGGS (Baldi et al. 2014), 11,000,000 rows"}

    def column(self, name):
        a = self.tbl.column(name)
        return a.to_numpy() if a.num_chunks == 1 else np.concatenate([c.to_numpy(zero_copy_only=False) for c in a.chunks])

    def target_values(self):
        return self.column(self.target)
