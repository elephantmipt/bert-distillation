from catalyst.core.callbacks import MetricCallback
import torch
from torch import nn


class PerplexityMetricCallbackDistillation(MetricCallback):
    """
    Perplexity is a very popular metric in NLP
    especially in Language Modeling task.
    It is 2^cross_entropy.
    """

    def __init__(
        self,
        input_key: str = "masked_lm_labels",
        output_key: str = "s_logits",
        prefix: str = "perplexity",
        ignore_index: int = None,
    ):
        """
        Args:
            input_key (str): input key to use for perplexity calculation,
                target tokens
            output_key (str): output key to use for perplexity calculation,
                logits of the predicted tokens
            ignore_index (int): index to ignore, usually pad_index
        """
        self.ignore_index = ignore_index or nn.CrossEntropyLoss().ignore_index
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index
        )
        super().__init__(
            metric_fn=self.metric_fn,
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
        )

    def metric_fn(
        self, s_logits: torch.Tensor, masked_lm_labels: torch.Tensor,
    ):
        """Calculate perplexity"""
        cross_entropy = self.cross_entropy_loss(
            s_logits.view(-1, s_logits.size(-1)), masked_lm_labels.view(-1),
        )
        perplexity = 2 ** cross_entropy
        return perplexity.item()
