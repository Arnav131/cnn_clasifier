# Instructions: Training and Evaluating the CNN + MOGEO + AdamW Pipeline on Google Colab

This document is the exact, step-by-step reference for running the pipeline in
`Colab_Training.ipynb`. Follow it in order. Every step maps to a numbered cell
in the notebook, so you can follow either document interchangeably.

---

## 0. What you are replacing

The previous pipeline (`MainCode.ipynb`) had several concrete, verified bugs:

- The `CNN_MOGE` optimization loop could not run at all (`NameError` on
  `y_train_resampled` in `evaluate_fitness`, and on `mutated_offspring` in
  `optimize`). Only a fixed-hyperparameter `Sequential` model was ever
  actually trained.
- `np.random.shuffle(X_train_indices)` and `np.random.shuffle(y_train)` were
  shuffled **independently**, decoupling images from their labels.
- Different cells used `test_size=0.1` and `test_size=0.2` with fresh calls to
  `train_test_split`, so "training" and "test" data were never a fixed,
  consistent partition -- silent train/test leakage.
- The "Performance" section generated accuracy/precision/recall/F1 via
  `np.random.uniform(0.98, 0.9989)` and sorted the random numbers to fake an
  improving trend. The "Comparison" charts hardcoded a "Proposed CbMOGE"
  score next to literature baselines -- none of it was measured.

None of the old numbers can be trusted. The new pipeline (`pipeline/`,
`run_baseline.py`, `run_mogeo.py`, `run_final.py`, `evaluate_final.py`,
`test_model.py`) fixes all of the above and only ever reports metrics from
real training/evaluation runs. `trainedd_cnn_model.h5` is preserved untouched
as the historical baseline artifact.

---

## 1. GPU setup

1. Open `Colab_Training.ipynb` in Google Colab (File > Open notebook > GitHub,
   paste `https://github.com/Arnav131/cnn_clasifier`, select the notebook --
   or upload it directly).
2. Go to **Runtime > Change runtime type**.
3. Set **Hardware accelerator** to **GPU** (T4 is sufficient; a faster GPU
   speeds up MOGEO search proportionally).
4. Click **Save**.
5. Run the first cell (`!nvidia-smi`). You should see a GPU listed. If you see
   an error, the runtime did not switch -- repeat step 2.

## 2. Mount Google Drive

Run the "Mount Google Drive" cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

A browser popup will ask you to authorize access to your Google account.
Approve it. This is required so that the dataset, split manifest, MOGEO
state, checkpoints, and results all persist across Colab disconnects --
nothing in this pipeline should live only on the ephemeral Colab VM disk.

## 3. Clone the repository (into Drive, not the VM disk)

Run the clone cell. It clones (or, on later sessions, `git pull`s)
`https://github.com/Arnav131/cnn_clasifier.git` into:

```
/content/drive/MyDrive/cnn_clasifier_project/cnn_clasifier
```

The notebook then auto-detects whether the pipeline code lives at the repo
root or inside a `code/` subfolder, and `%cd`s into the correct `PROJECT_DIR`.
All subsequent commands in this document assume your shell's current
directory is that `PROJECT_DIR` (the folder directly containing `dataset/`,
`pipeline/`, `run_baseline.py`, etc.).

**Why clone into Drive?** If you clone into `/content/` (the VM's local
disk), everything is deleted the moment the Colab runtime disconnects or
recycles. Cloning into Drive means dataset + code + every checkpoint you
generate survive across sessions, and resuming after a disconnect is just
"reopen the notebook, re-run the cells from the top."

## 4. Install dependencies

```bash
pip install -q -r requirements-colab.txt
```

This installs `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, `tqdm`.

**Do not** `pip install torch torchvision` on Colab. The Colab GPU runtime
already ships CUDA-enabled builds of both; reinstalling from PyPI risks
overwriting them with a CPU-only build or a mismatched CUDA version. The
notebook's dependency cell asserts `torch.cuda.is_available()` right after
installing -- if that assertion fails, your runtime is not actually using a
GPU (revisit step 1), it is not a dependency problem.

## 5. Dataset verification

Run the dataset verification cell. It walks `dataset/`, counts class folders
and images per class, and asserts:

- exactly 150 class folders exist,
- the total image count is in a sane range (> 4000).

If either assertion fails, the clone/Drive sync did not finish -- wait for
Drive to finish syncing (check the Drive web UI) and re-run the clone cell.

## 6. Build the train/dev/test split manifest + run leakage checks

```python
from pipeline.data import build_manifest
from pipeline.utils import leakage_report

df = build_manifest('dataset', 'splits')
print(df['split'].value_counts())
print(leakage_report(df.to_dict('records')))
```

This is the step that fixes the original pipeline's biggest flaw. It:

1. Performs **one** stratified split of the full dataset: 15% held out as
   `final_test`, 85% kept as the development set.
2. Splits the development set again (80/20, stratified) into `dev_train`
   (~68% of the total data) and `dev_val` (~17% of the total data).
3. Hashes every image (MD5) and detects byte-identical duplicates. If a
   duplicate pair spans an evaluation split (`dev_val` or `final_test`) and
   any other split, **all** copies are reassigned into `dev_train` -- this
   prevents a duplicate image from silently leaking training information
   into a split that is supposed to measure generalization. The details are
   written to `splits/duplicate_report.txt`.
4. Writes `splits/manifest.csv` (one row per image: filepath, label,
   class_idx, split, file_hash) and `splits/classes.json`.

This manifest is written **once** and is idempotent -- calling
`build_manifest` again just loads the cached CSV. Every script below
(`run_baseline.py`, `run_mogeo.py`, `run_final.py`, `evaluate_final.py`,
`test_model.py`) reads this exact same file, so `final_test` is guaranteed
to be the identical 15% everywhere, and it is only ever fed to a model in
step 10 below.

`leakage_report(...)` explicitly asserts that no filepath appears in more
than one split and prints `RESULT: PASS` or raises with the exact
overlapping files if something is wrong -- do not proceed past a `FAIL`.

## 7. Baseline training

```bash
python run_baseline.py --dataset-dir dataset --output-dir . --resume
```

Trains a fixed-hyperparameter CNN (3 conv blocks, 32 base channels --
deliberately close in spirit to the historical architecture) with **AdamW**,
on `dev_train`, early-stopping on `dev_val` accuracy (patience 5, up to 20
epochs). This isolates "fixing the pipeline + switching to AdamW" from
"adding MOGEO search," so you can honestly attribute later gains to MOGEO.

Writes:
- `checkpoints/baseline_best.pt`, `checkpoints/baseline_last.pt` (resume state)
- `results/baseline_training_curves.png`
- a row in `results/experiment_log.csv` (`stage=baseline`)

**If Colab disconnects**, just re-run the exact same command (`--resume` is
already in it). It loads `checkpoints/baseline_last.pt` and continues from
the next epoch instead of restarting.

## 8. MOGEO hyperparameter/architecture search

```bash
python run_mogeo.py --dataset-dir dataset --output-dir . --pop-size 8 --generations 6 --resume
```

This is the **outer meta-optimizer**. It is a genuine Multi-Objective Golden
Eagle Optimizer (`pipeline/mogeo.py`) -- not random search, not grid search:

- Each "eagle" is a point in a 7-dimensional hyperparameter space (`num_blocks`,
  `base_channels`, `dropout`, `lr`, `weight_decay`, `batch_size`,
  `label_smoothing`).
- Eagles move via **attack vectors** (toward a Pareto-optimal "prey" position
  drawn from the archive by crowding-distance tournament) and **cruise
  vectors** (a random direction orthogonalized against the attack vector via
  Gram-Schmidt, modeling circling flight). The balance between attack and
  cruise shifts from exploratory to exploitative as generations progress.
- Multi-objective handling uses NSGA-II-style fast non-dominated sorting and
  crowding distance to maintain a Pareto archive across two genuinely
  competing objectives: internal validation **accuracy** and internal
  validation **macro-F1** (both measured on `dev_val` only).
- Each candidate hyperparameter vector is evaluated by actually training a
  CNN with **AdamW** for ~10 epochs (`config.MOGEO_CANDIDATE_EPOCHS`) on
  `dev_train` and scoring it on `dev_val`. `final_test` is never touched.

Default budget: population=8, generations=6 (up to 48 short trainings). This
is a compute/time trade-off for a single Colab GPU session -- increase
`--pop-size`/`--generations` if you have more time available; the algorithm
itself does not change.

Writes (all under `results/` and `checkpoints/`):
- `mogeo_results.csv` -- one row per evaluated candidate (generation, eagle
  index, decoded hyperparameters, `val_acc`, `val_macro_f1`)
- `experiment_log.csv` -- appended with `stage=mogeo` rows
- `mogeo_convergence.png` -- best-so-far accuracy/F1 per generation
- `best_hparams.json` -- the selected "best compromise" solution from the
  final Pareto archive (the archive member maximizing the mean of the two
  objectives)
- `checkpoints/mogeo_state.pkl` -- full resumable optimizer state

**Resumability (important):** state is saved after **every single candidate
evaluation**, not just at the end of a generation. If Colab disconnects mid-
search, re-running the exact same command continues from the last completed
candidate -- at most one partially-evaluated candidate is repeated.

**Do not delete `checkpoints/mogeo_state.pkl` between runs unless you want to
start the search over.** If you want to restart the search from scratch,
delete that file first (running without `--resume` while it exists will
print a warning and abort, precisely to stop you from accidentally starting a
second, conflicting search).

## 9. Final training

```bash
python run_final.py --dataset-dir dataset --output-dir . --resume
```

Loads `results/best_hparams.json` (falls back to the baseline hyperparameters
with a printed warning if MOGEO has not been run) and retrains on
`dev_train`/`dev_val` with **AdamW**, using:

- a hard cap of **40 epochs** (`--epochs`, clamped server-side to 40 even if
  you pass more -- the spec is explicit that this must not be forced blindly),
- early stopping **patience = 7** on validation accuracy.

Training stops as soon as it plateaus; it will very likely stop well before
40 epochs. Writes:

- `checkpoints/best.pt` -- **the** final deliverable checkpoint (best
  validation-accuracy epoch, model weights + hyperparameters)
- `checkpoints/final_last.pt` -- resume state (safe to delete after training)
- `results/training_curves.png`
- an `experiment_log.csv` row with `stage=final`

**If Colab disconnects**, re-run the same command; `--resume` continues from
`checkpoints/final_last.pt`.

## 10. Final evaluation -- exactly once, on the untouched final_test split

```bash
python evaluate_final.py --dataset-dir dataset --output-dir .
```

This is the **only** script in the entire pipeline that reads the
`final_test` split. Run it once, after step 9 has finished. It loads
`checkpoints/best.pt`, runs inference on the 15% held-out test images, and
writes:

- `results/accuracy_report.md` -- human-readable summary (accuracy, macro
  precision/recall/F1, weighted F1, and an explicit note distinguishing this
  report from the old fabricated "Performance" section)
- `results/accuracy_report.json` -- the same metrics as structured data
- `results/confusion_matrix.png`
- `results/error_analysis.md` -- worst-performing classes by recall, and the
  most common true-vs-predicted confusion pairs with example file paths

Re-running this script is harmless (it is read-only with respect to
`final_test`), but you should treat the **first** run's numbers as your
reported result -- repeatedly tweaking hyperparameters based on `final_test`
feedback and re-running this script would reintroduce the leakage this whole
pipeline was built to avoid. If you want to iterate further, go back to step
8/9 and only ever look at `dev_val` metrics while doing so.

## 11. Download `best.pt` and results back to your machine

Run the download cell in the notebook:

```python
import shutil
from google.colab import files

shutil.make_archive('cnn_clasifier_deliverables', 'zip', '.', 'checkpoints')
shutil.make_archive('cnn_clasifier_deliverables_results', 'zip', '.', 'results')
files.download('cnn_clasifier_deliverables.zip')
files.download('cnn_clasifier_deliverables_results.zip')
```

Two zip files will download through your browser:
- `cnn_clasifier_deliverables.zip` -- contains `checkpoints/` (including `best.pt`)
- `cnn_clasifier_deliverables_results.zip` -- contains `results/` (all reports/plots/csvs)

Everything also remains on Google Drive at
`/content/drive/MyDrive/cnn_clasifier_project/cnn_clasifier/` for future sessions.

## 12. Feed `best.pt` back into Antigravity / your local repo

1. Unzip `cnn_clasifier_deliverables.zip` locally.
2. Copy `best.pt` into your local clone of this repository, at
   `checkpoints/best.pt` (create the folder if it does not exist).
3. Also copy `splits/manifest.csv` and `splits/classes.json` from Drive (or
   regenerate them locally with `build_manifest`, using the same seed --
   `config.SEED = 42` -- to get an identical split) so `test_model.py` can map
   class indices back to plant names.
4. Open this repository in Antigravity/Zed and continue from there -- e.g.
   ask the agent to inspect `results/accuracy_report.md` and
   `results/error_analysis.md` for follow-up analysis, or to help design the
   next MOGEO search iteration.

## 13. Testing `best.pt` with `test_model.py`

Locally (or in the same Colab session), from the `PROJECT_DIR`:

```bash
# Classify a single image and look up its medicinal properties
python test_model.py --checkpoint checkpoints/best.pt --image "Amaranthus viridis (1).jpg"

# Classify every image in a folder
python test_model.py --checkpoint checkpoints/best.pt --dir path/to/some/folder

# Recompute accuracy on the untouched final_test split (quick sanity check;
# for the full report use evaluate_final.py instead)
python test_model.py --checkpoint checkpoints/best.pt --eval-test --dataset-dir dataset
```

`test_model.py` loads `splits/classes.json` to map predicted indices back to
plant names, and (if present) looks up `properties.csv` to print the
botanical name, family, medicinal property, and side effects of the
predicted class, mirroring the original notebook's Tkinter viewer without
its GUI dependency.

---

## Appendix: file/folder map

| Path | Purpose |
|---|---|
| `pipeline/config.py` | Paths, seeds, split fractions, epoch budgets, `HParams` |
| `pipeline/data.py` | Manifest builder (85/15 split, dedup/leakage fix), `Dataset`/`DataLoader` |
| `pipeline/model.py` | `ConfigurableCNN` (architecture driven by `HParams`) |
| `pipeline/mogeo.py` | Multi-Objective Golden Eagle Optimizer |
| `pipeline/train.py` | AdamW training loop, resumable checkpointing, early stopping |
| `pipeline/evaluate.py` | Metrics, confusion matrix, curves, MOGEO convergence, reports |
| `run_baseline.py` | Baseline experiment entrypoint |
| `run_mogeo.py` | MOGEO search entrypoint |
| `run_final.py` | Final training entrypoint (produces `best.pt`) |
| `evaluate_final.py` | One-time `final_test` evaluation entrypoint |
| `test_model.py` | Ad hoc / single-image / folder testing of `best.pt` |
| `splits/manifest.csv`, `splits/classes.json` | Fixed dev/test partition (generated) |
| `results/` | All reports, plots, and CSV logs (generated) |
| `checkpoints/` | `best.pt`, resume state, MOGEO state (generated) |
| `MainCode.ipynb`, `trainedd_cnn_model.h5` | Legacy notebook + historical baseline model -- preserved, unused by the new pipeline |
| `extracted_features.csv`, `preprocessed_dataset/`, `training_data/`, `testing_data/` | Legacy classical-CV feature extraction artifacts -- **not used** by the new CNN pipeline |

## Appendix: honesty checklist for reported results

Before trusting/publishing any number produced by this pipeline, confirm:

- [ ] `splits/manifest.csv` was built once and never rebuilt with
      `force_rebuild=True` mid-experiment (rebuilding reshuffles the split).
- [ ] `leakage_report(...)` printed `RESULT: PASS` before training started.
- [ ] `evaluate_final.py` was run exactly once against the final `best.pt`,
      and no hyperparameter changes were made afterward based on its output.
- [ ] `results/accuracy_report.md`'s numbers came from that single run --
      compare its "Notes on validity" section, which explicitly documents
      this constraint.
