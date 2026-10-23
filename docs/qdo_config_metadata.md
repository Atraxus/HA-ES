# QDO config metadata: hyperparameters namespaced per model family

Branch `qdo-config-metadata` changes the per-config dict that `QDO` compares in configuration space. The dict now matches the format of the original phem/Q(D)O-ES. This page explains:

- what that dict is used for;
- what was wrong with it;
- what it looks like now;
- which results the change affects.

## What the dict is used for

phem's QDO-ES places every ensemble in a 2-D behavior space. Each method in `src/generate_data.py` uses a different space:

| Method | Behavior space | Reads `model_metadata["config"]` |
| --- | --- | --- |
| `QDO` | config space Gower similarity × loss correlation | **yes** |
| `ENS_SIZE_QDO` | ensemble size × loss correlation | no |
| `INFER_TIME_QDO`, `MEMORY_QDO`, `DISK_QDO` | loss correlation × summed inference time / memory / disk | no |
| `QO` | none (quality-only archive) | no |

The hardware-aware methods are built with `base_models_metadata_type="custom"`. For those methods, phem hands each member's whole `model_metadata` to the behavior function, which reads only `val_predict_time`, `memory` or `diskspace`. phem computes ranges over `config` only for `QDO`.

phem's `config_space_gower_similarity` averages over all pairs of members in an ensemble. For one pair, it averages `1 - distance` over the keys the two dicts **share**:

- a string value counts as categorical: distance 0 when equal, 1 when not;
- any other value is numeric: `|a - b|` divided by the key's range over the whole pool.

Configs that share more keys, with closer values, look more alike. 1 means no diversity and 0 means maximal diversity.

## What original phem puts in the dict

In the original Q(D)O-ES code (`LennartPurucker/PopulationBasedQDO-PostHocEnsembleSelectionAutoML`, `assembled_ask/ask_assembler.py`), `config` is the Auto-sklearn pipeline configuration (`bm_model.config._values`):

- **Keys are namespaced by component**, for example `classifier:random_forest:max_features`, and `classifier:__choice__` names the model. Two different classifiers therefore share only `classifier:__choice__`, which differs, plus pipeline-level keys such as balancing and preprocessing. Those are genuinely the same hyperparameters.
- **Categoricals are kept as strings**, and numeric ranges skip string values.

## What was wrong

Before this branch, `load_and_process_base_models` used TabArena's raw hyperparameter names. It kept only values that `float()` accepts and added `model_type`. That diverged from phem in four ways. The numbers below are phem's Gower similarity, averaged over 400 random pairs per family pair of the cached TabArena configs.

| # | Problem | Since | Effect |
| --- | --- | --- | --- |
| 1 | **Keys are not namespaced.** Different families reuse names: `learning_rate` (CatBoost, LightGBM, NeuralNetTorch, XGBoost), `subsample` and `colsample_bylevel` (CatBoost, XGBoost), `max_cat_to_onehot` (LightGBM, XGBoost), and `bootstrap`, `max_features`, `min_impurity_decrease`, `min_samples_split`, `n_estimators` (RandomForest, ExtraTrees). EBM also has `learning_rate` and `cat_smooth`. | TabRepo version, so it is in the published QDO numbers | Configs of different families looked partly alike: RF×XT 0.59, CAT×XGB 0.52, GBM×XGB 0.47, and the families that share `learning_rate` with NN_TORCH about 0.35. With namespaced keys, all of these are 0. |
| 2 | **Ranges are pooled across families.** A shared key's range spans every family that has it, which squeezes the distances within a family. | Follows from 1 | RF within-family similarity was 0.81 instead of 0.75. |
| 3 | **`ag.verbosity` is treated as a hyperparameter.** Every config of the 2026-07-13 suite carries `ag.verbosity: 2` (CatBoost, ExtraTrees, probably EBM). `include_ag_args=False` removes only `ag_args`, and `2` passes the float filter. | TabArena port | CAT×XT scored 0.50 instead of 0. Within-family similarity rose for CatBoost (0.79 instead of 0.74) and ExtraTrees (0.89 instead of 0.86). |
| 4 | **Categoricals are dropped.** Examples: CatBoost and XGBoost `grow_policy`, NeuralNetTorch `activation`, NeuralNetFastAI `layers`, and ExtraTrees `max_features="sqrt"`. | TabRepo version | Affects within-family similarity only: ExtraTrees 0.86→0.78, NeuralNetFastAI 0.73→0.64, NeuralNetTorch 0.73→0.70, XGBoost 0.75→0.73, CatBoost 0.74→0.76. LightGBM and RandomForest are about unchanged. |

Problems 1–3 matter most when the pool mixes families. They distort the diversity axis of `QDO`: an RF+XT ensemble looked much less diverse than an RF+FastAI one, although both span two families.

## What the dict looks like now

`config_metadata` and `categorical_if_mixed` in `src/tabarena_data.py` build the dict as follows:

- `model_type` names the family, and every hyperparameter key is namespaced as `<model_type>:<key>`;
- keys starting with `ag_args` or `ag.` (AutoGluon bookkeeping) are skipped;
- numbers, including bools, become floats, and everything else (strings, lists, None) becomes a string;
- a key whose values mix numbers and strings across the pool, such as ExtraTrees `max_features` (0.5 or `"sqrt"`), is compared as categorical. phem would otherwise subtract a string from a number. This is the same rule EnsembleGym uses.

Configs of different families now share only `model_type`, so their similarity is 0, and each numeric range covers a single family.

Before and after, for one RF and one XT config:

```python
# RandomForest_r1_BAG_L1
before = {"bootstrap": 1.0, "max_features": 0.767, "max_samples": 0.908, "min_impurity_decrease": 4.59e-05,
          "min_samples_split": 2.0, "n_estimators": 50.0, "model_type": "RF"}
after  = {"model_type": "RF", "RF:bootstrap": 1.0, "RF:max_features": 0.767, "RF:max_samples": 0.908,
          "RF:min_impurity_decrease": 4.59e-05, "RF:min_samples_split": 2.0, "RF:n_estimators": 50.0}

# ExtraTrees_r73_default_BAG_L1
before = {"bootstrap": 0.0, "min_impurity_decrease": 1e-05, "min_samples_split": 26.0, "n_estimators": 50.0,
          "ag.verbosity": 2.0, "model_type": "XT"}
after  = {"model_type": "XT", "XT:bootstrap": 0.0, "XT:max_features": "sqrt", "XT:min_impurity_decrease": 1e-05,
          "XT:min_samples_split": 26.0, "XT:n_estimators": 50.0}
```

Before, these two configs shared five hyperparameter keys. Now they share only `model_type`, which differs.

## Default configs

TabArena stores each family's default config (`*_c1_*`) with no hyperparameters (`{}`), meaning "use AutoGluon's defaults". Its dict therefore holds only `model_type`. As a result, it has similarity 1 to every config of its own family and 0 to every other family. The previous dict gave the same values, with one exception: the CatBoost and ExtraTrees defaults also shared `ag.verbosity` with each other's family, so those pairs scored 0.5 instead of 0. Original phem never has this case, because Auto-sklearn configs always carry values.

This branch leaves the default config as it is. Filling in the real defaults would need each model library's default values, some of which depend on the problem type. For scale: in EnsembleGym's cross-family runs, 12% (pruned pool) and 23% (full pool) of the QDO-ES ensembles contain a default config.

## Relation to EnsembleGym

EnsembleGym (`ADA-research/EnsembleGym`) already uses this format. Its cross-family loader (`family_metadata` in `src/dataloaders/tabarena_cross_family.py`) namespaces keys and adds `model_type`. Its QDO-ES (`model_hyperparameters` in `src/algorithms/qdo/qdo_es.py`) skips `ag_args*` and `ag.*` keys, keeps categoricals, and compares mixed keys as categorical. The QDO metadata of the two repositories is now the same.

There is one difference. EnsembleGym's single-family pools carry no `model_type`, so a default config shares no key with the other configs and its pairs are left out of the mean.

## Which results change

- **`QDO` results are not comparable across this change.** Rerun `QDO` before comparing it with the other methods.
- **No other method changes**: `SINGLE_BEST`, `GES`, `MULTI_GES`, `QO`, `ENS_SIZE_QDO`, `INFER_TIME_QDO`, `MEMORY_QDO` and `DISK_QDO` do not read the dict.
- There is no flag to restore the old dict. The published QDO numbers were computed on TabRepo, and problem 3 is specific to TabArena, so a legacy flag could not reproduce them anyway.
