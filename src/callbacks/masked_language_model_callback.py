from typing import Dict, List, Union

from catalyst.core import MetricCallback
import torch
from torch import nn


class MaskedLanguageModelCallback(MetricCallback):
    """
    Callback to compute masked language model loss
    """

    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = None,
        output_key: Union[str, List[str], Dict[str, str]] = None,
        prefix: str = "masked_lm_loss",
        multiplier: float = 1.0,
        **metric_kwargs,
    ):
        """
        Args:
            input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole input will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole output will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            prefix (str): prefix for metrics and output key for loss
                in ``state.batch_metrics`` dictionary
            criterion_key (str): A key to take a criterion in case
                there are several of them and they are in a dictionary format.
            multiplier (float): scale factor for the output loss.
        """
        if output_key is None:
            output_key = "s_logits"
        if input_key is None:
            input_key = "masked_lm_labels"
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            metric_fn=self.metric_fn,
            **metric_kwargs,
        )
        self._criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def metric_fn(
        self, s_logits: torch.Tensor, masked_lm_labels: torch.Tensor
    ):
        """
        Computes cross-entropy loss on given logits
        Args:
            s_logits: tensor shape of (batch_size, seq_len, voc_len)
            masked_lm_labels: tensor shape of (batch_size, seq_len)

        Returns:
            cross-entropy loss
        """
        loss_mlm = self._criterion(
            s_logits.view(-1, s_logits.size(-1)), masked_lm_labels.view(-1),
        )
        return loss_mlm
