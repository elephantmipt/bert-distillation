from typing import Dict, List, Union

from catalyst.core import MetricCallback
import torch
from torch import nn


class MSELossCallback(MetricCallback):
    """Callback to compute MSE loss"""

    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = None,
        output_key: Union[str, List[str], Dict[str, str]] = None,
        prefix: str = "mse_loss",
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
            multiplier (float): scale factor for the output loss.
        """
        if output_key is None:
            output_key = [
                "t_logits",
                "s_logits",
                "attention_mask",
            ]
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            metric_fn=self.metric_fn,
            **metric_kwargs,
        )
        self._criterion = nn.MSELoss(reduction="mean")

    def metric_fn(
        self,
        t_logits: torch.Tensor,
        s_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Computes MSE loss for given distributions
        Args:
            s_logits: tensor shape of (batch_size, seq_len, voc_size)
            t_logits: tensor shape of (batch_size, seq_len, voc_size)
            attention_mask:  tensor shape of (batch_size, seq_len, voc_size)
        Returns:
            MSE loss
        """
        mask = attention_mask.unsqueeze(-1).expand_as(s_logits)
        # (bs, seq_lenth, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)
        # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
        # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)
        # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))
        # (bs * seq_length, voc_size) modulo the 1s in mask
        loss_mse = self._criterion(s_logits_slct, t_logits_slct)
        return loss_mse
