from typing import Dict

from catalyst import dl
from catalyst.dl.utils import is_wrapped_with_ddp
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistilMLMRunner(dl.Runner):
    """Simplified huggingface Distiller wrapped with catalyst"""

    def __init__(
        self,
        alpha_kl: float = 0.75,
        alpha_mlm: float = 0.05,
        alpha_mse: float = 0.0,
        alpha_cos: float = 0.2,
        *runner_args,
        **kwargs,
    ):
        """Init runner.
        Args:
            alpha_kl: coefficient for KL from
                student to teacher network
            alpha_mlm: coefficient for Cross-Entropy loss mlm
            alpha_mse: coefficient for MSE between hidden states
            alpha_cos: coefficient for Cosine loss between hidden states
        """
        super().__init__(*runner_args, **kwargs)
        self.alpha_kl = alpha_kl
        self.alpha_mlm = alpha_mlm
        self.alpha_mse = alpha_mse
        self.alpha_cos = alpha_cos
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

    def _handle_batch(self, batch: Dict[str, torch.Tensor]):
        if is_wrapped_with_ddp(self.model):
            teacher, student = (
                self.model.module["teacher"],
                self.model.module["student"],
            )
        else:
            teacher, student = self.model["teacher"], self.model["student"]

        teacher.eval()  # manually set teacher model to eval mode
        attention_mask = batch["input_ids"] != 0
        with torch.no_grad():
            t_logits, t_hidden_states = teacher(
                batch["input_ids"], attention_mask
            )

        s_logits, s_hidden_states = student(batch["input_ids"], attention_mask)

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

        loss = 0
        if self.alpha_kl > 0.0:
            loss_kl = self.kl_loss_fct(
                F.log_softmax(s_logits_slct, dim=-1),
                F.softmax(t_logits_slct, dim=-1),
            )
            loss += self.alpha_kl * loss_kl
            self.state.batch_metrics["loss_kl"] = loss_kl

        if self.alpha_mlm > 0.0:
            loss_mlm = self.lm_loss_fct(
                s_logits.view(-1, s_logits.size(-1)),
                batch["masked_lm_labels"].view(-1),
            )
            loss += self.alpha_mlm * loss_mlm
            self.state.batch_metrics["loss_mlm"] = loss_mlm

        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(
                s_logits_slct, t_logits_slct
            ) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse
            self.state.batch_metrics["loss_mse"] = loss_mse

        if self.alpha_cos > 0.0:
            loss_cos = self._loss_cos(
                s_hidden_states, t_hidden_states, batch["attention_mask"]
            )
            loss += self.alpha_cos * loss_cos
            self.state.batch_metrics["loss_cos"] = loss_cos

        self.state.batch_metrics["loss"] = loss

    def _loss_cos(
        self,
        s_hidden_states: torch.Tensor,
        t_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        loss_cos = self.cosine_loss_fct(
            s_hidden_states_slct, t_hidden_states_slct, target
        )
        return loss_cos
