from torch import nn
import transformers


class BertForMLM(nn.Module):
    """
    BertForMLM

    Simplified huggingface model
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_logits: bool = True,
        output_hidden_states: bool = True,
    ):
        """
        Args:
            model_name: huggingface model name
            output_logits: same as in huggingface
            output_hidden_states: same as in huggingface
        """
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states,
            output_logits=output_logits,
        )
        self.bert = transformers.BertForMaskedLM.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, *model_args, **model_kwargs):
        """Forward method"""
        return self.bert(*model_args, **model_kwargs)
