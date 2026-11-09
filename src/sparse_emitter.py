# Sparse emitter for QDO-ES, from "MO-GES: Multi-Objective Greedy Ensemble Selection".
# The file is unchanged below this header.
#
from __future__ import annotations

import numpy as np

from phem.methods.ensemble_selection.qdo.emitters import DiscreteWeightSpaceEmitter


class SparseDiscreteWeightSpaceEmitter(DiscreteWeightSpaceEmitter):
    """DiscreteWeightSpaceEmitter whose mutation can also remove a base model from the ensemble.

    Similar to the reset in MO-GES: a mutation step can set the count of an ensemble member to zero instead of
    increasing the count of a base model. Removal steps are only taken if the ensemble has at least two members.
    Crossover, selection, and ask/tell are unchanged.

    Parameters
    ----------
    fixed_sparse_probability: bool, default=False
        If True, a mutation removes an ensemble member with probability `removal_probability`.
        If False, removal steps are selected uniformly together with the remaining increase steps of the weight
        vector (similar to MO-GES, where removal and addition candidates compete equally).
    removal_probability: float, default=0.2
        Probability that a mutation removes an ensemble member (if a removal is possible).
        Only used if fixed_sparse_probability is True.
    """

    def __init__(
        self,
        *args,
        fixed_sparse_probability: bool = False,
        removal_probability: float = 0.2,
        **kwargs,
    ):
        if not 0 <= removal_probability <= 1:
            raise ValueError(f"removal_probability must be in [0, 1]. Got: {removal_probability}")

        super().__init__(*args, **kwargs)
        self.fixed_sparse_probability = fixed_sparse_probability
        self.removal_probability = removal_probability
        self.n_removals = 0

        # Members not yet removed per weight vector (removing the same member always yields the same vector)
        self._remaining_removals_for_hash = {}

    def _step_for_weights(self, discrete_weight_vector):
        """Remove an ensemble member or take a default step (see fixed_sparse_probability).

        Returns:
            new discrete weight vector
            step distance, negative for a removal such that the new sample size is computed correctly.
        """
        dwv_hash = hash(tuple(discrete_weight_vector))
        if dwv_hash not in self._remaining_removals_for_hash:
            members = np.flatnonzero(discrete_weight_vector)
            self._remaining_removals_for_hash[dwv_hash] = list(members) if len(members) >= 2 else []
        remaining_removals = self._remaining_removals_for_hash[dwv_hash]

        if remaining_removals:
            if self.fixed_sparse_probability:
                p_remove = self.removal_probability
            else:
                # Uniform selection over remaining removal and increase steps
                if dwv_hash not in self._remaining_steps_for_hash:
                    self._init_step_lookup(discrete_weight_vector, dwv_hash)
                n_increase_steps = len(self._remaining_steps_for_hash[dwv_hash])
                p_remove = len(remaining_removals) / (len(remaining_removals) + n_increase_steps)

        if remaining_removals and (self._rng.random() < p_remove):
            self.n_removals += 1
            remove_idx = remaining_removals.pop(self._rng.integers(len(remaining_removals)))

            tmp_dmw = discrete_weight_vector.copy()
            step_distance = -abs(int(round(tmp_dmw[remove_idx])))
            tmp_dmw[remove_idx] = 0
            return tmp_dmw, step_distance

        return super()._step_for_weights(discrete_weight_vector)
