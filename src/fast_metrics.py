"""Fast drop-in replacement for phem's ROC AUC metric.

phem's ROC AUC (a copy of scikit-learn's) validates its inputs on every call. Ensemble selection
calls it hundreds of thousands of times per task on inputs that are already valid, so most of the
time goes into that validation. This module performs the same computation (building the ROC
curve and integrating it with the trapezoidal rule) with the same numpy operations in the same
order, but without the validation. The results are bit-for-bit identical, which matters because
ensemble selection compares losses of candidate ensembles directly, so even differences in the
last bit can change which ensemble is selected.
"""

import numpy as np
from phem.base_utils.metrics import AbstractMetric


def _binary_roc_auc(y_true_bool: np.ndarray, y_score: np.ndarray) -> float:
    """Same computation as `phem.base_utils.custom_metrics.roc_auc._binary_roc_auc_score`."""
    n_pos = np.count_nonzero(y_true_bool)
    if n_pos == 0 or n_pos == len(y_true_bool):
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined.")

    # _binary_clf_curve: sort by decreasing score and count true/false positives per threshold
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true_bool[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true * 1.0, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # roc_curve: drop collinear points and start the curve at (0, 0)
    if len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # auc: fpr is increasing
    return np.trapz(tpr, fpr)


class FastRocAuc(AbstractMetric):
    """ROC AUC as returned by `phem.application_utils.supported_metrics.msc("roc_auc", ...)`.

    Binary: AUC of the positive class (label 1). Multiclass: unweighted mean of the one-vs-rest
    AUCs over the `labels` present in `y_true`. Apart from rejecting non-finite scores, inputs are
    not validated.
    """

    def __init__(self, is_binary: bool, labels: list[int]):
        super().__init__(
            metric=self._roc_auc,
            name="roc_auc",
            maximize=True,
            classification=True,
            transform_conf_to_pred=False,
            optimum_value=1,
            pos_label=1,
            requires_confidences=True,
            only_positive_class=is_binary,
        )
        self.is_binary = is_binary
        self.labels = list(labels)

    def _roc_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        # The only input check of phem's metric that can fail for valid ensemble predictions
        if not np.isfinite(y_score).all():
            raise ValueError("Input contains NaN or infinity.")
        if self.is_binary:
            return float(_binary_roc_auc(y_true == self.pos_label, y_score))

        # One-vs-rest; like phem, classes that do not occur in y_true are left out
        present = np.unique(y_true)
        scores = np.array(
            [
                _binary_roc_auc(y_true == label, y_score[:, i])
                for i, label in enumerate(self.labels)
                if label in present
            ]
        )
        return float(np.mean(scores))

    def __call__(self, y_true, y_pred, to_loss: bool = False, checks: bool = False):
        return super().__call__(y_true, y_pred, to_loss=to_loss, checks=checks)
