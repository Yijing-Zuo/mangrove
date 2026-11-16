# Mangrove

Batched GPU training of many histogram GBDT models with per-model results identical to training each model alone.

## Dependencies

Our code was tested under the following dependencies:

- Linux, NVIDIA A100 40GB / A100 80GB / H100 80GB / RTX PRO 6000 Blackwell / L4
- Python 3.13
- CUDA 13.0 (driver 580.82), PyTorch 2.14.1, Triton 3.8.0
- NumPy 2.4.6, scikit-learn 1.6.1, openml 0.15.1
- XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10
- OpenBoost 1.0.0.dev0 (commit 5ac60ee, its own environment with numpy 2.3.5, numba 0.63.1, numba-cuda 0.27.0, cupy-cuda13x 14.2.0)

Data preparation was tested with NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, openml 0.15.1 (`requirements-data.txt`).

```
pip install -r requirements.txt
MG_ARCHS=80 bash cuda/build.sh cuda
python -m scripts.prepare_data
```

`MG_ARCHS` is the GPU's compute capability (80 A100, 90 H100, 120 RTX PRO 6000, 89 L4). `prepare_data` downloads the tables from OpenML, prepares them as in the paper and checks every array against the recorded hashes; `python -m scripts.prepare_data adult higgs-50k` prepares only some tables.

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

Each study runs its workloads in order, writes one result file per run to `out/<study>/` and the end-to-end time of every run to `out/<study>/timing.jsonl`. `--print` lists the commands; `--tables` restricts the tables. The XGBoost pool runs start the CUDA MPS daemon; every other run stops it.

Table 3 and Figure 6, with and without sampling (10 tables x 4 procedures x 3 seeds, sequential Hyperband on Adult and Higgs-50K):

```
python -m scripts.run sampling
python -m scripts.run no-sampling
python -m scripts.summarize searches out/sampling
python -m scripts.summarize searches out/no-sampling
python -m scripts.check alone out/no-sampling/higgs-1m/sha_mangrove_s0.json --k 6
```

Table 4, other baselines and settings, for `{extra-tables, depths, settings, lightgbm-catboost, cpu, openboost, breadth}`:

```
python -m scripts.run extra-tables
python -m scripts.summarize searches out/extra-tables
python -m scripts.run cpu
python -m scripts.summarize fixed out/cpu
```

Other GPUs: the same `sampling` and `no-sampling` commands after building with that GPU's `MG_ARCHS`. OpenBoost runs with the interpreter of its own environment.

Figure 1 (XGBoost process pools), Figure 7 (learning over groups), Figure 10 (number of models) and Figure 11a (number of rows):

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

Figure 11b (Higgs-1M to Higgs-11M, one A100 80GB, 60 GiB budget):

```
python -m scripts.run large
python -m scripts.summarize searches out/large
```

Figure 8 (round anatomy), Figure 9 (ablation), Figure 12 and Section 6.6 (memory budgets), Figure 5 (lifecycle trace):

```
python -m scripts.run anatomy
python -m scripts.summarize anatomy out/anatomy
python -m scripts.run ablation
python -m scripts.summarize ablation out/ablation
python -m scripts.run budgets
python -m scripts.summarize memory out/budgets
python -m scripts.run trace
python -m scripts.summarize memory out/trace
```

Speedups are ratios of medians over three seeds, summarized by geometric means over workloads.

## License

MIT (see `LICENSE`).
