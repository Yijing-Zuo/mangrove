# Mangrove

Batched GPU training of many histogram GBDT models; each model's trees and margins are identical to training it alone in Mangrove on the same GPU.

## Dependencies

Our code was tested under the following dependencies:

- Linux, NVIDIA A100 40GB / A100 80GB / H100 80GB / RTX PRO 6000 Blackwell / L4
- Python 3.13
- CUDA 13.0 (driver 580.82), PyTorch 2.14.1, Triton 3.8.0
- NumPy 2.4.6, scikit-learn 1.6.1, openml 0.15.1
- XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, CuPy 14.0.1 (cupy-cuda13x)

Data preparation was tested with NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, openml 0.15.1 (`requirements-data.txt`).

```
pip install -r requirements.txt
MG_ARCHS=80 bash cuda/build.sh cuda
python -m scripts.prepare_data
```

`MG_ARCHS` is the GPU's compute capability (80 A100, 90 H100, 120 RTX PRO 6000, 89 L4). `prepare_data` downloads the tables from OpenML (Year from the UCI repository; Higgs-2M to Higgs-11M from a Parquet mirror or the UCI HIGGS file), prepares them as in the paper and checks every array against the recorded hashes; `python -m scripts.prepare_data adult higgs-50k` prepares only some tables.

## Usage

Quick checks: every line of a block prints the same fingerprint, and the squared-loss fingerprints match across the two. The emulation build and test run the kernels on the CPU.

```
python -m scripts.check cpu
python -m scripts.check gpu
python -m pytest -q tests/test_gpu.py
python -m scripts.check full
bash cuda/build.sh emu
MANGROVE_EMU=1 python -m pytest -q tests/test_emu.py
```

Each study runs its workloads in order, writes one result file per run to `out/<study>/` and the end-to-end time of every run to `out/<study>/timing.jsonl`. `--print` lists the commands; `--tables` restricts the tables. Every XGBoost run on the GPU starts the CUDA MPS daemon, except a single process with 12 threads; every other run stops it.

Section 6.2, Table 4 (with and without sampling), Table 5 and Figure 5 (with sampling) and the quality results (10 tables x 4 procedures x 3 seeds, sequential Hyperband on Adult and Higgs-50K):

```
python -m scripts.run sampling
python -m scripts.run no-sampling
python -m scripts.summarize searches out/sampling
python -m scripts.summarize searches out/no-sampling
python -m scripts.summarize breakdown out/sampling
```

Section 6.1, the 16-process pool on eight searches without sampling, against the seed-0 XGBoost pool searches of `no-sampling`:

```
python -m scripts.run pool16
```

Section 6.2, the tuned pool against Mangrove and the XGBoost pool on the 42 workloads of `sampling` (with sampling):

```
python -m scripts.run tuned
python -m scripts.summarize searches out/tuned xgb_gpu_tuned_8x1
python -m scripts.summarize searches out/tuned xgb_gpu_8x1 xgb_gpu_tuned_8x1
```

Section 6.2, Table 6 (without sampling) and the CatBoost comparison in the text, for `{extra-tables, depths, settings, lightgbm-catboost, cpu, breadth}`:

```
python -m scripts.run extra-tables
python -m scripts.summarize searches out/extra-tables
python -m scripts.run lightgbm-catboost
python -m scripts.summarize searches out/lightgbm-catboost lgb_cpu_12x1
python -m scripts.summarize searches out/lightgbm-catboost catboost_gpu_1x1
python -m scripts.summarize fixed out/lightgbm-catboost lgb_cpu
python -m scripts.run cpu
python -m scripts.summarize fixed out/cpu
```

Table 6, other GPUs (without sampling): build with that GPU's `MG_ARCHS`, then run `python -m scripts.run no-sampling` (on the L4, add `--tables airlines,sgemm,medical-charges,particulate-matter`) and `python -m scripts.summarize searches out/no-sampling`.

Section 2.3 (XGBoost process pools), Section 6.3 (learning over groups, Figure 6) and Section 6.5 (number of models and 50K to 1M rows, Figure 9a):

```
python -m scripts.run pools
python -m scripts.summarize pools out/pools
python -m scripts.run groups
python -m scripts.summarize fixed out/groups
python -m scripts.run models
python -m scripts.summarize models out/models
python -m scripts.run rows
python -m scripts.summarize fixed out/rows
```

Section 6.5, Figure 9b, 1M to 11M rows (Higgs-1M to Higgs-11M, one A100 80GB, 60 GiB budget), against each XGBoost setup (in the paper, eight processes did not finish Higgs-11M within three hours):

```
python -m scripts.run large
python -m scripts.summarize searches out/large xgb_gpu_8x1
python -m scripts.summarize searches out/large xgb_gpu_4x3
python -m scripts.summarize searches out/large xgb_gpu_1x12
```

Section 6.4 (round anatomy and ablation, Figures 7 and 8), Section 6.6 (memory budgets, Figure 10, whose 16 GiB points are the seed-0 searches of `no-sampling`) and Section 5.2 (lifecycle trace, Figure 4):

```
python -m scripts.run anatomy
python -m scripts.summarize anatomy out/anatomy
python -m scripts.run ablation
python -m scripts.summarize ablation out/ablation
python -m scripts.run ablation-no-sampling
python -m scripts.summarize ablation out/ablation-no-sampling
python -m scripts.run budgets
python -m scripts.summarize memory out/budgets
python -m scripts.run trace
python -m scripts.summarize memory out/trace
```

Section 6.6, equivalence to training alone: six models of each of nine searches without sampling, retrained alone:

```
python -m scripts.check alone out/no-sampling/adult/hyperband-sequential_mangrove_s0.json out/no-sampling/airlines/sha_mangrove_s0.json out/no-sampling/sgemm/asha_mangrove_s0.json out/no-sampling/covertype/sha_mangrove_s0.json out/no-sampling/covertype/hyperband-parallel_mangrove_s0.json out/no-sampling/higgs-1m/sha_mangrove_s0.json out/no-sampling/higgs-1m/asha_mangrove_s0.json out/no-sampling/year/hyperband-parallel_mangrove_s0.json out/depths/higgs-1m/sha_mangrove_s0.json --k 6
```

Speedup is a baseline's time divided by Mangrove's, each the median over three seeds (three runs of a fixed set). The comparisons marked with a dagger in Table 6 (with the exceptions in its note), the CatBoost searches, the 1M to 11M-row searches and the budget study use one run (seed 0 for searches). Averages over workloads are geometric means.

## License

MIT (see `LICENSE`).
