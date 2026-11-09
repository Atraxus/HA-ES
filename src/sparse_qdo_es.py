# Adapted from the RAAML implementation for "MO-GES: Multi-Objective Greedy Ensemble Selection".
"""QDO-ES with RAAML's sparse mutation (`infer-qdo-es-sparse`).

RAAML's `QDOEnsembleSelection(sparse_mutation=True)` differs from phem's only in the emitter class,
which `_init_emitters` builds. `_init_emitters` below is phem's, with RAAML's change.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
from phem.methods.ensemble_selection.qdo.qdo_es import QDOEnsembleSelection

from sparse_emitter import SparseDiscreteWeightSpaceEmitter


class SparseQDOEnsembleSelection(QDOEnsembleSelection):
    """QDO-ES whose mutation can also remove an ensemble member.

    Parameters
    ----------
    fixed_sparse_probability: bool, default=False
        Passed to SparseDiscreteWeightSpaceEmitter. The default is RAAML's default.
    All other parameters are those of phem's QDOEnsembleSelection.
    """

    def __init__(self, *args, fixed_sparse_probability: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_sparse_probability = fixed_sparse_probability

    def _init_emitters(self, predictions, y_true):
        _start_weight_vectors = self._init_start_weight_vectors(predictions, y_true)

        # Code for analysis
        if self.show_analysis:
            o, b = self._evaluate_batch_of_solutions(
                np.array([w for w, _ in _start_weight_vectors]),
                predictions,
                y_true,
            )
            self._sb_stats = [o[np.argmax(o)], b[np.argmax(o), :]]

        # -- Build emitters
        if self.emitter_method == "DiscreteWeightSpaceEmitter":
            for req_time_atr in ["mutation_probability_after_crossover", "crossover_probability"]:
                if (req_time_atr in self.emitter_vars) and (
                    isinstance(self.emitter_vars[req_time_atr], Callable)
                ):
                    self.emitter_vars[req_time_atr] = partial(
                        self.emitter_vars[req_time_atr],
                        max_time=self.internal_n_iterations,
                    )

            emitters = [
                SparseDiscreteWeightSpaceEmitter(
                    self.archive,
                    self.n_base_models,
                    _start_weight_vectors,
                    batch_size=self.batch_size,
                    seed=self.random_state.randint(1, 100000),
                    fixed_sparse_probability=self.fixed_sparse_probability,
                    **self.emitter_vars,
                ),
            ]
            self.discrete_merge_ = True
        else:
            raise RuntimeError("You should not be here!")

        self.emitters = emitters

        # Set proba normalization if needed
        #   If negative weights are proposed, we need to normalize probabilities
        #   for some metrics. Here, we set a flag to do so later.
        metrics_need_proba = self.score_metric.requires_confidences or (
            "proba" in self.behavior_space.required_prediction_types
        )
        if any(emt.proposes_negative_weights for emt in self.emitters) and metrics_need_proba:
            self.normalize_predict_proba_ = True
