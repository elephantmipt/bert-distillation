from typing import Dict, List, Union

from catalyst.core import MetricCallback
import torch
from torch import nn


class CosineLossCallback(MetricCallback):
    """
    CosineLossCallback
    This callback is calculating cosine loss between hidden states
    of the two hugging face transformers models.
    """

    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = None,
        output_key: Union[str, List[str], Dict[str, str]] = None,
        prefix: str = "cosine_loss",
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
            output_key = [
                "t_hidden_states",
                "s_hidden_states",
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
        self._criterion = nn.CosineEmbeddingLoss(reduction="mean")

    def metric_fn(
        self,
        t_hidden_states: torch.Tensor,
        s_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes cosine loss on given hidden states
        Args:
            t_hidden_states: tensor from teacher model
            s_hidden_states: tensor from student model
            attention_mask: tensor with attention mask on given batch

        Returns:
            cosine loss
        """
        s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
        t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
        mask = attention_mask.unsqueeze(-1).expand_as(
            s_hidden_states
        )  # (bs, seq_length, dim)
        assert s_hidden_states.size() == t_hidden_states.size()
        dim = s_hidden_states.size(-1)

        s_hidden_states_slct = torch.masked_select(
            s_hidden_states, mask
        )  # (bs * seq_length * dim)
        s_hidden_states_slct = s_hidden_states_slct.view(
            -1, dim
        )  # (bs * seq_length, dim)
        t_hidden_states_slct = torch.masked_select(
            t_hidden_states, mask
        )  # (bs * seq_length * dim)
        t_hidden_states_slct = t_hidden_states_slct.view(
            -1, dim
        )  # (bs * seq_length, dim)

        target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(
            1
        )  # (bs * seq_length,)
        loss_cos = self._criterion(
            s_hidden_states_slct, t_hidden_states_slct, target
        )
        return loss_cos
