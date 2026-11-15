"""phem's QDO behavior descriptors, from pairwise values computed once per task.

phem recomputes two descriptors for every ensemble a QDO search evaluates:

- the average loss correlation (``LossCorrelationMeasure``): the per-row loss of each member on
  every validation row, then ``np.corrcoef`` over the members
- the config-space Gower similarity (``ConfigSpaceGowerSimilarity``): a Python loop over all pairs
  of members

Both are means over pairs of members, so each pair's value can be computed once for the pool and an
ensemble's descriptor read from those values. On a full pool this was 80-90% of a QDO search's
time. The idea is EnsembleGym's (``src/algorithms/qdo/behavior.py``); this module builds it from
phem's own functions so that every descriptor equals phem's bit for bit:

- loss correlation: the losses come from phem's ``not_aggregated_loss`` with phem's label encoding,
  and an ensemble's value is ``np.mean`` over the upper triangle of its members' block of
  ``nan_to_num(|corrcoef|)``, the order phem uses. Bit-identity also needs BLAS to compute a pair's
  correlation the same in the pool's matrix as in an ensemble's; it does with the single-threaded
  OpenBLAS the jobs use, and each search checks its first ensembles against phem (``N_CHECKS``).
- Gower similarity: each pair's value is phem's ``config_space_gower_similarity`` of that pair,
  with the ranges phem computes over the pool, and an ensemble's value is ``statistics.mean`` over
  its pairs as in phem. ``statistics.mean`` is exactly rounded, so this is identical by construction.

A one-member ensemble gets phem's value, 1. The searches therefore take the same path as with phem's
descriptors and return the same ensembles.
"""

from __future__ import annotations

import statistics
from itertools import combinations

import numpy as np
from phem.base_utils.diversity_metrics import LossCorrelation, not_aggregated_loss
from phem.methods.ensemble_selection.qdo.behavior_functions.basic import LossCorrelationMeasure
from phem.methods.ensemble_selection.qdo.behavior_functions.implicit_diversity_metrics import (
    ConfigSpaceGowerSimilarity,
    config_space_gower_similarity,
)
from phem.methods.ensemble_selection.qdo.behavior_space import BehaviorFunction, BehaviorSpace
from phem.methods.ensemble_selection.qdo.qdo_es import QDOEnsembleSelection
from sklearn.preprocessing import LabelBinarizer

# Ensembles of two or more members per search whose descriptors are also computed by phem and must
# match exactly
N_CHECKS = 20


def _transformed_labels(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """The one-hot labels of phem's ``average_loss_correlation``."""
    lb = LabelBinarizer()
    lb.fit(y_true)
    if len(lb.classes_) != n_classes:
        lb.classes_ = list(range(n_classes))
    transformed_labels = lb.transform(y_true)
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels, transformed_labels, axis=1)
    return transformed_labels


def loss_correlation_matrix(predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """``nan_to_num(|corrcoef|)`` of the per-row losses of every pair of base models."""
    transformed_labels = _transformed_labels(y_true, predictions[0].shape[-1])
    losses = np.array([not_aggregated_loss(transformed_labels, bm) for bm in predictions])
    # A constant loss gives NaN, which phem counts as no correlation
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(np.abs(np.corrcoef(losses)))


def gower_inputs(base_models, metric) -> tuple[dict[str, float], list[dict]]:
    """The config ranges over the pool and the configs, as phem's QDO-ES computes them."""
    search = QDOEnsembleSelection(
        base_models=base_models,
        n_iterations=1,
        score_metric=metric,
        behavior_space=BehaviorSpace([ConfigSpaceGowerSimilarity]),
        n_jobs=1,
    )
    return search.config_key_to_range_, search.qdo_base_models_metadata_


def gower_similarity_pairs(ranges: dict[str, float], configs: list[dict]) -> list[list]:
    """phem's Gower similarity of every pair of configs: ``pairs[i][j]`` for ``i < j``."""
    n = len(configs)
    return [
        [None] * (i + 1)
        + [config_space_gower_similarity((ranges, [configs[i], configs[j]])) for j in range(i + 1, n)]
        for i in range(n)
    ]


class _LossCorrelation:
    """One search's loss correlation descriptor."""

    def __init__(self, matrix: np.ndarray, predictions: np.ndarray, y_true: np.ndarray) -> None:
        self.matrix = matrix
        self.predictions = predictions
        self.y_true = y_true
        self.checks_left = N_CHECKS

    def __call__(self, weights: np.ndarray):
        members = np.flatnonzero(weights)
        if len(members) == 0:
            raise ValueError("An ensemble without members has no loss correlation")
        if len(members) == 1:
            return LossCorrelation.single_model_ensemble_default
        rows, cols = np.triu_indices(len(members), k=1)
        value = np.mean(self.matrix[members[rows], members[cols]])
        if self.checks_left:
            self.checks_left -= 1
            expected = LossCorrelation(
                self.y_true, [self.predictions[i] for i in members], checks=False
            )
            if value != expected:
                raise RuntimeError(
                    f"Precomputed loss correlation {value!r} differs from phem's {expected!r} "
                    f"for members {members.tolist()}"
                )
        return value


class _GowerSimilarity:
    """One search's config-space Gower similarity descriptor."""

    def __init__(self, pairs: list[list], ranges: dict[str, float], configs: list[dict]) -> None:
        self.pairs = pairs
        self.ranges = ranges
        self.configs = configs
        self.checks_left = N_CHECKS

    def __call__(self, weights: np.ndarray):
        members = np.flatnonzero(weights).tolist()
        if len(members) == 0:
            raise ValueError("An ensemble without members has no Gower similarity")
        if len(members) == 1:
            # phem's value for a single config
            return 1
        value = statistics.mean([self.pairs[i][j] for i, j in combinations(members, 2)])
        if self.checks_left:
            self.checks_left -= 1
            expected = config_space_gower_similarity(
                (self.ranges, [self.configs[i] for i in members])
            )
            if value != expected:
                raise RuntimeError(
                    f"Precomputed Gower similarity {value!r} differs from phem's {expected!r} "
                    f"for members {members}"
                )
        return value


class PrecomputedBehaviors:
    """A task's pairwise descriptor values, shared by all its QDO searches.

    `base_models` and `metric` are needed only for the Gower similarity; without them a behavior
    space keeps phem's own Gower similarity.
    """

    def __init__(self, predictions: np.ndarray, y_true: np.ndarray, base_models=None, metric=None):
        self.predictions = predictions
        self.y_true = y_true
        self.loss_correlation = loss_correlation_matrix(predictions, y_true)
        self.gower = None
        if base_models is not None:
            ranges, configs = gower_inputs(base_models, metric)
            self.gower = (gower_similarity_pairs(ranges, configs), ranges, configs)

    def apply(self, behavior_space: BehaviorSpace) -> BehaviorSpace:
        """`behavior_space` with phem's pairwise descriptors read from this task's values.

        Each call makes new descriptor objects, so every search checks its own first ensembles.
        """
        functions = []
        for bf in behavior_space.behavior_functions:
            if bf is LossCorrelationMeasure:
                # Declared "proba" like phem's, so the search prepares the same inputs as before
                function = _LossCorrelation(self.loss_correlation, self.predictions, self.y_true)
                bf = BehaviorFunction(function, ["weights"], bf.range_tuple, "proba", name=bf.name)
            elif bf is ConfigSpaceGowerSimilarity and self.gower is not None:
                function = _GowerSimilarity(*self.gower)
                bf = BehaviorFunction(function, ["weights"], bf.range_tuple, "none", name=bf.name)
            functions.append(bf)
        return BehaviorSpace(functions)
