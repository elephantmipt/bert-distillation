from typing import Dict

from catalyst import dl
from catalyst.utils import is_wrapped_with_ddp
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistilMLMRunner(dl.Runner):
    """Simplified huggingface Distiller wrapped with catalyst"""

    def __init__(
        self,
        alpha_kl: float = 0.95,
        alpha_mlm: float = 0.05,
        *runner_args,
        **kwargs,
    ):
        """Init runner.
        Args:
            alpha_kl: coefficient for KL from
                student to teacher network
            alpha_mlm: coefficient for Cross-Entropy loss mlm
        """
        super().__init__(*runner_args, **kwargs)
        self.alpha_kl = alpha_kl
        self.alpha_mlm = alpha_mlm
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def _handle_batch(self, batch: Dict[str, torch.Tensor]):
        if is_wrapped_with_ddp(self.model):
            teacher, student = (
                self.model.module["teacher"],
                self.model.module["student"],
            )
        else:
            teacher, student = self.model["teacher"], self.model["student"]

        teacher.eval()
        with torch.no_grad():
            t_logits, t_hidden_state = \
                teacher(batch["features"], batch["attention_mask"])

        student.train()
        s_logits, s_hidden_states = \
            student(batch["features"], batch["attention_mask"])
        mask = batch["attention_mask"].unsqueeze(-1).expand_as(s_logits)
        # (bs, seq_lenth, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)
        # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
        # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)
        # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))
        # (bs * seq_length, voc_size) modulo the 1s in mask

        loss_kl = self.kl_loss_fct(
            F.log_softmax(s_logits_slct, dim=-1),
            F.softmax(t_logits_slct, dim=-1),
        )
        loss_mlm = self.lm_loss_fct(
            s_logits.view(-1, s_logits.size(-1)), batch["mlm_labels"].view(-1)
        )

        loss = self.alpha_kl * loss_kl + self.alpha_mlm * loss_mlm
        self.state.batch_metrics = {
            "loss_kl": loss_kl,
            "loss_mlm": loss_mlm,
            "loss": loss,
        }

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()
