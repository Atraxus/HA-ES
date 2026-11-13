# HAPEns: Hardware-Aware Post-Hoc Ensembling
Ensembling is commonly used in machine learning on tabular data to boost predictive performance and robustness, but larger ensembles often lead to increased hardware demand. HAPEns is a post-hoc ensembling method that explicitly balances accuracy against hardware efficiency. Inspired by multi-objective and quality diversity optimization, HAPEns constructs a diverse set of ensembles along the Pareto front of predictive performance and resource usage, from which practitioners can select the ensemble that fits their deployment constraints. HAPEns extends our earlier work on hardware-aware ensemble selection (HA-ES).

HAPEns maintains a population of ensembles in a two-dimensional behavior space spanned by the average loss correlation of the ensemble's base models and a hardware cost metric (memory usage by default). The space is divided into a 7x7 sliding boundaries archive, in which each niche keeps its best ensemble. New ensembles are created from the archive by crossover and mutation. The hardware cost of an ensemble is the sum of the costs of all base models with non-zero weight.

To evaluate HAPEns and compare it to the baselines, we use [TabArena](https://github.com/autogluon/tabarena), which provides cached validation and test prediction probabilities of many tuned model configurations on its curated datasets. We use this data to efficiently evaluate and compare the post-hoc ensembling methods without training any models.

The experiments in the publications below were run on TabRepo, the predecessor of TabArena, with the `D244_F3_C1530_100` context (83 classification datasets, 10 seeds). The TabRepo-based code is available in the git history (commit `e3fab45`). TabArena contains fewer, curated datasets (38 classification datasets) and different model configurations, so results obtained with this version are not directly comparable to the published ones.

## Methods
`src/generate_data.py` evaluates the following methods on the ROC AUC. Results are stored under their ID.

| ID | Name in the paper | Method |
| --- | --- | --- |
| `MEMORY_QDO` | HAPEns | Quality diversity optimization over average loss correlation and memory usage |
| `SINGLE_BEST` | Single-Best | The single base model with the best validation score |
| `GES` | GES* | Greedy ensemble selection, returning the ensemble of every iteration |
| `MULTI_GES-<w>` | Multi-GES(`w`) | GES with a static weighting `w` of inference time against predictive performance; the paper uses `w = 0.68` |
| `QDO` | QDO-ES | Quality diversity optimization over average loss correlation and config space similarity (not hardware-aware). The similarity compares hyperparameters within a model family; configs of different families count as maximally dissimilar (see `docs/qdo_config_metadata.md`) |
| `INFER_TIME_QDO`, `DISK_QDO`, `ENS_SIZE_QDO` | Inference Time, Diskspace, Ensemble Size | Ablations of HAPEns with a different cost metric |
| `QO` | | Quality optimization without a behavior space (not used in the paper) |

The Pareto fronts of the methods are compared with the hypervolume (HV) and IGD+ in `notebooks/plotting.ipynb`, based on the test ROC AUC and the cost metrics normalized per task and seed.

## Set-Up
This set-up guide expects a Linux system. The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.11 or 3.12.

### Get the code
`phem` is a git submodule in `extern/`. If you did not clone the repository with `--recurse-submodules`, initialize it from the project root:
- `git submodule update --init --recursive`

### Dependencies
From the project root, create the virtual environment in `.venv` and install the locked dependencies (TabArena, AutoGluon and `phem`):
- `uv sync`

To also install the dependencies of the notebooks, use `uv sync --all-groups`.

### TabArena data
The base models are the configs of the TabArena methods listed in `DEFAULT_METHODS` in `src/tabarena_data.py`. Their processed artifacts (predictions, labels and metrics) are stored in `~/.cache/tabarena` (override with the `TABARENA_CACHE` environment variable). Each method is several GB, so select a subset with `--methods` for a smaller run. Download them once before running the experiments, in particular before starting several runs in parallel:
- `uv run python src/download_data.py`

The memory and disk usage of each config is not part of TabArena. It is measured on dummy data by fitting every config with AutoGluon, which needs the model libraries of the `measure` dependency group:
- `uv sync --group measure`
- `uv run python src/config_stats.py`

which writes `data/model_memory_and_disk_usage.csv`. The dummy data is seeded, so runs are reproducible, but keep the CPU count fixed between runs you compare: AutoGluon sizes the models' thread pools by it, which changes the inference memory of some models. Configs without a measurement fall back to the median of their model type. Without the file, `generate_data.py` skips Memory-QDO and Disk-QDO and runs the other methods. The `Dockerfile` runs the same script in a container.

The memory cost of a config is the increase in resident memory of a fresh process from loading the fitted model and predicting, measured after a warm-up load so that library code shared by all ensemble members is not attributed to each model (`src/memory_probe.py`, Linux only). Each config is fitted with the model class TabArena benchmarked it with.

### Run
To run the experiments for one seed use
- `uv run python src/generate_data.py --seed 0`

Several seeds (`--seed 0 1 2`) are fitted on each task after loading it once. Results that already exist are skipped, so delete them to recompute.

For a quick test, restrict the methods, datasets and folds, e.g.
- `uv run python src/generate_data.py --methods ExtraTrees --folds 0 --datasets blood-transfusion-service-center`

The ensembles of a task are fitted in parallel on all available CPUs (set the number with `--n-jobs`). A task consists of 27 ensemble fits per seed, 20 of which (GES and Multi-GES) take similarly long, so with one seed 10 or 20 CPUs are used most efficiently, and with three seeds 20 or 30. Multi-GES with weight 0 is GES and is not fitted, except with `--legacy-multi-ges`.

In the code used for the publications, the Multi-GES weights were fitted one after another with the same ensemble object, which limited each weight to the best iteration of the previous weight instead of 100 iterations. This is fixed; `--legacy-multi-ges` reproduces the previous behavior.

`bash multiple_runs_job.sh` runs seeds 0-9 on a Slurm cluster (arguments such as `--seed` or `--folds` are passed to `generate_data.py`). It submits `single_run_job.sh` as one array with an element per classification dataset and repeat of `--folds` (TabArena's splits 0-2 are repeat 0, 3-5 repeat 1, ...), which fits all seeds. Every element loads TabArena's metadata and labels, while each task's predictions are read by one element only, once for all seeds; for shorter jobs, `--split-folds` gives each fold its own element and `--seeds-per-job N` splits the seeds into groups of N, at the cost of loading the metadata, and a task's predictions, once per element. Set the time limit with e.g. `SBATCH_TIMELIMIT=06:00:00`. A resubmission reruns only the missing results. Afterwards, `check_results.py` lists missing result files and `src/process_data.py` aggregates the results into `data/`, which the notebooks in `notebooks/` use for plotting.

### Inference time against performance on pruned pools
`src/inference_pareto.py` compares Infer-QDO-ES (full budget, 3 batches, and sparse with a static removal probability of 20%), GES, QO-ES, QDO-ES and each family's best config on validation, by test Elo against the ensemble's inference time. It runs on two pools, the 9 classic methods and the 12 foundation models, each pruned with TabArena's `EnsembleScorerMaxModels`, on the classification datasets and with ROC AUC. Every search gets 40 steps times the pool size (`--steps`), except the 3-batch variant, which always gets the single models plus 3 batches of 40. Each Infer-QDO-ES variant enters at 3 positions along its validation front, from the fastest ensemble to the best on validation (`aggregate --front-points`). TabArena's `compare` scores all entrants in one run, with RandomForest's default at Elo 1000, so Elos are only comparable within one `aggregate` and with the same number of front points.

It needs the plotting dependencies and the methods of `PARETO_POOLS` in `src/tabarena_data.py`. No memory or disk measurements are needed. To run seed 0 on a Slurm cluster (one array element per dataset and repeat that fits all seeds, then a job that merges them and draws the plot), run this from the project root:
- `uv sync --group notebooks`
- replace `YOUR_PARTITION` in `inference_pareto_job.sh` with your Slurm partition
- `export TABARENA_CACHE=/path/to/tabarena/store`
- `bash inference_pareto_job.sh`

For one dataset without Slurm, use `uv run python src/inference_pareto.py run --datasets blood-transfusion-service-center` and then `uv run python src/inference_pareto.py aggregate`. Results are written to `results/inference_pareto/`, the points and leaderboard to `data/inference_pareto_*.csv`, and the figure to `plots/inference-pareto-mean-time.png`. `data/` and `plots/` are not tracked; the figure and both CSVs of the 10-seed run on all 38 datasets (`aggregate --seed 0 1 2 3 4 5 6 7 8 9 --name ""`) are kept in `docs/inference_pareto/`, together with a version without the full-budget Infer-QDO-ES (`aggregate --seed 0 1 2 3 4 5 6 7 8 9 --searches infer_qdo_3_batches infer_qdo_sparse_p20 --name no_infer_qdo`). `--searches` leaves the other searches out of the `compare` run, so the Elos differ slightly from the run with all searches.

For several seeds, use `bash inference_pareto_job.sh --seed 0 1 2 3 4 5 6 7 8 9`, or `aggregate --seed 0 1 2` on finished runs. Each task is loaded, pruned and scored for the family rows once for all seeds, and seeds whose results exist are skipped. Results are written per repeat (`seed_<S>/<dataset>/repeat_<R>/`); move results of the earlier per-dataset layout (`seed_<S>/<dataset>/`) into `repeat_0/` before mixing them with new runs. Each seed's fronts are built separately, then each entrant's error and inference time are averaged per task over the seeds and scored in one `compare` run. All seeds must cover the same tasks. The outputs get the suffix `_<n>_seeds` (`-<n>-seeds` for the figure), so they do not replace a single seed's.

#### Raw results of the 10-seed run
The raw results behind `docs/inference_pareto/` (seeds 0-9, all 38 classification datasets, folds 0-2, both pools) are kept in `docs/inference_pareto/inference_pareto_results.zip` (12.5 MB). Extract it into `results/` from the project root:
- `unzip docs/inference_pareto/inference_pareto_results.zip -d results`

which creates `results/inference_pareto/seed_<S>/<dataset>/repeat_0/` with two files per task:
- `ensembles.parquet`: one row per ensemble a search returned. GES, QO-ES and QDO-ES return one ensemble per fold; each Infer-QDO-ES variant returns every distinct ensemble of its final archive, of which `aggregate` keeps the validation Pareto front. Besides the task (`pool`, `dataset`, `fold`, `problem_type`, `metric`, `pool_size`) and the search (`algorithm`, `ensemble` as the index within it), a row stores the test and validation error (`metric_error`, `metric_error_val`, 1 − ROC AUC), the inference time (`time_infer_s`, the sum of the members' TabArena inference times), and the ensemble itself (`n_members`, `members` as TabArena config names, `weights`).
- `families.parquet`: the config of each model family in the pool with the best validation error, in TabArena's result format.

`aggregate` reads only these files, so the leaderboards and figures can be recomputed without the TabArena data, e.g. `uv run python src/inference_pareto.py aggregate --seed 0 1 2 3 4 5 6 7 8 9 --name ""`.

## Relevant Publications
If you use HAPEns or HA-ES in scientific publications, we would appreciate citations.

Maier, J., & Purucker, L. (2026). HAPEns: Hardware-Aware Post-Hoc Ensembling for Tabular Data. arXiv. https://arxiv.org/abs/2603.10582

Maier, J., Möller, F., & Purucker, L. (2024). Hardware Aware Ensemble Selection for Balancing Predictive Accuracy and Cost. Paper presented at the Third International Conference on Automated Machine Learning (AutoML 2024) Workshop. arXiv. https://arxiv.org/abs/2408.02280


I have also written my Master's thesis on this topic: _Hardware-Aware Ensemble Selection for Balancing Predictive Accuracy and Operational Costs in AutoML Systems_

The thesis goes into much more detail and introduces some new methods. I would be happy to provide it to anyone interested. 
