from torchmetrics import BinnedPrecisionRecallCurve
from typing import Any, Dict, List, Tuple, Union
from torch import Tensor
import torch

DEFAULT_JSON = '''{
"training_set": "Ramasamy22/ST",
"test_set": "default",
"test_fold": 0,
"save_model": false,

"representation":"ESM1_small",
"freeze_representation": true,

"receptive_field": 65,
"conv_depth":3,
"conv_width":9,
"conv_channels": 200,
"final_fc_neurons":64,
"dropout":0.2,
"max_pool_size":1,
"batch_norm":false,

"batch_size":4,
"learning_rate":1e-4,
"pos_weight":1,
"max_epochs":10
}'''

def _precision_at_recall(
    precision: Tensor,
    recall: Tensor,
    thresholds: Tensor,
    min_recall: float,
) -> Tuple[Tensor, Tensor]:
    """
    Quick but suboptimal solution to get precision-at-fixed-recall. Adapted from TorchMetrics code for
    BinnedRecallAtFixedPrecision. See TorchMetrics documentation for more details
    """
    try:
        max_precision, _, best_threshold = max(
            (p, r, t) for p, r, t in zip(precision, recall, thresholds) if r >= min_recall
        )

    except ValueError:
        max_precision = torch.tensor(0.0, device=precision.device, dtype=precision.dtype)
        best_threshold = torch.tensor(0)

    if max_precision == 0.0:
        best_threshold = torch.tensor(1e6, device=thresholds.device, dtype=thresholds.dtype)

    return max_precision, best_threshold

class BinnedPrecisionAtFixedRecall(BinnedPrecisionRecallCurve):
    """
    Quick but suboptimal solution to get precision-at-fixed-recall. Adapted from TorchMetrics code for
    BinnedRecallAtFixedPrecision. See TorchMetrics documentation for more details
    """
    def __init__(
        self,
        num_classes: int,
        min_recall: float,
        thresholds: Union[int, Tensor, List[float]] = 100,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(num_classes=num_classes, thresholds=thresholds, **kwargs)
        self.min_recall = min_recall

    def compute(self) -> Tuple[Tensor, Tensor]:  # type: ignore
        """Returns float tensor of size n_classes."""
        precisions, recalls, thresholds = super().compute()

        if self.num_classes == 1:
            return _precision_at_recall(precisions, recalls, thresholds, self.min_recall)

        precisions_at_p = torch.zeros(self.num_classes, device=recalls[0].device, dtype=recalls[0].dtype)
        thresholds_at_p = torch.zeros(self.num_classes, device=thresholds[0].device, dtype=thresholds[0].dtype)
        for i in range(self.num_classes):
            precisions_at_p[i], thresholds_at_p[i] = _precision_at_recall(
                precisions[i], recalls[i], thresholds[i], self.min_recall
            )
        return precisions_at_p, thresholds_at_p

def get_gpu_max_batchsize(representation: str, freeze_representation: bool):
    """
    Manually set an upper limit to the batch size to be used on your system. This depends on the representation type.
    This is not optimized, so please modify this to fit your own system.

    Parameters
    ----------
    representation : str
        The used protein language model

    freeze_representation : bool
        If True, the language model weights will not be fine-tuned, and thus less GPU memory is needed
    """
    if representation == 'onehot':
        return 64
    elif representation == 'ESM1_small':
        if freeze_representation:
            return 64
        else:
            return 8
    elif representation == 'ESM1b':
        if freeze_representation:
            return 16
        else:
            return 4
    else:
        return 4