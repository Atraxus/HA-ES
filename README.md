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
| `QDO` | QDO-ES | Quality diversity optimization over average loss correlation and config space similarity (not hardware-aware) |
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

The memory and disk usage of each config is not part of TabArena. It is measured on dummy data by
- `uv run python src/config_stats.py`

which writes `data/model_memory_and_disk_usage.csv`. Configs without a measurement fall back to the median of their model type. The `Dockerfile` runs the same script in a container.

### Run
To run the experiments for one seed use
- `uv run python src/generate_data.py --seed 0`

For a quick test, restrict the methods, datasets and folds, e.g.
- `uv run python src/generate_data.py --methods ExtraTrees --folds 0 --datasets blood-transfusion-service-center`

`single_run_job.sh` and `multiple_runs_job.sh` run seeds 0-9 on a Slurm cluster. Afterwards, `check_results.py` lists missing result files and `src/process_data.py` aggregates the results into `data/`, which the notebooks in `notebooks/` use for plotting.

## Relevant Publications
If you use HAPEns or HA-ES in scientific publications, we would appreciate citations.

Maier, J., & Purucker, L. (2026). HAPEns: Hardware-Aware Post-Hoc Ensembling for Tabular Data. arXiv. https://arxiv.org/abs/2603.10582

Maier, J., Möller, F., & Purucker, L. (2024). Hardware Aware Ensemble Selection for Balancing Predictive Accuracy and Cost. Paper presented at the Third International Conference on Automated Machine Learning (AutoML 2024) Workshop. arXiv. https://arxiv.org/abs/2408.02280


I have also written my Master's thesis on this topic: _Hardware-Aware Ensemble Selection for Balancing Predictive Accuracy and Operational Costs in AutoML Systems_

The thesis goes into much more detail and introduces some new methods. I would be happy to provide it to anyone interested. 
