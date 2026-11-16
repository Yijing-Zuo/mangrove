import os
import pytest
from util import ROOT, synth, sample_hpo, make_specs, obs
from mangrove import Engine, Cohort

EMU_LIB = ROOT / "cuda" / "libmangrove_emu.so"
pytestmark = pytest.mark.skipif(os.environ.get("MANGROVE_EMU") != "1" or not EMU_LIB.exists(),
                                reason="needs MANGROVE_EMU=1 and cuda/libmangrove_emu.so")


def test_gpu_backend_emulation_matches_ref():
    import mangrove.kernels_gpu as KG
    KG.EMU = True
    Xb, y, folds = synth(1500, 14, nb=32, seed=0, task="binary")
    loss = "logistic"
    cfgs = sample_hpo(3, 4, dmin=2, dmax=6, sub=(0.7, 1.0), col=(0.5, 1.0))
    specs = make_specs(cfgs, folds, n_rounds=3, fold_list=(0, 2))
    for i, s in enumerate(specs):
        s.quant_bits = 28 if i % 2 else 16
    co = [Cohort("g", Xb, y)]
    a = Engine(co, loss); ia = a.add_models(specs); a.run()
    g = Engine(co, loss, backend="gpu"); ig = g.add_models(specs); g.run()
    assert g.KR is KG
    assert all(obs(a, x) == obs(g, z) for x, z in zip(ia, ig))
