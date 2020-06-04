from typing import Dict
from collections import OrderedDict

from catalyst import dl
from catalyst.dl.utils import is_wrapped_with_ddp
import torch


class DistilMLMRunner(dl.Runner):
    """Simplified huggingface Distiller wrapped with catalyst"""

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

        self.state.output = OrderedDict()
        self.state.output["attention_mask"] = attention_mask
        self.state.output["t_hidden_states"] = t_hidden_states
        self.state.output["s_hidden_states"] = s_hidden_states
        self.state.output["s_logits"] = s_logits
        self.state.output["t_logits"] = t_logits
