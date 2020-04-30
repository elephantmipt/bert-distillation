from typing import Dict, List

import torch
from torch import nn
import transformers


class DistilbertStudentModel(nn.Module):
    """
    DistilbertStudentModel

    Distil model class based on huggingface class but with
    initialization in it. Model will take vocabulary
    layers from specified teacher model
    """

    def __init__(
        self, teacher_model_name: str = "bert-base-uncased", layers=None
    ):
        """
        Args:
            teacher_model_name: name of the model to distil
            layers: layers indexes to initialize
        """
        super().__init__()
        if layers is None:
            layers = [0, 2, 4, 7, 9, 11]
        teacher_config = transformers.AutoConfig.from_pretrained(
            teacher_model_name, output_hidden_states=True, output_logits=True
        )
        teacher = transformers.BertForMaskedLM.from_pretrained(
            teacher_model_name, config=teacher_config
        )
        distil_sd = self._extract(teacher, layers)
        if teacher_model_name == "bert-base-uncased":
            student_config = transformers.AutoConfig.from_pretrained(
                "distilbert-base-uncased",
                output_hidden_states=True,
                output_logits=True,
            )
            self.student = transformers.DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-uncased",
                config=student_config,
                state_dict=distil_sd,
            )
        elif teacher_model_name == "bert-base-cased":
            student_config = transformers.AutoConfig.from_pretrained(
                "distilbert-base-cased",
                output_hidden_states=True,
                output_logits=True,
            )
            self.student = transformers.DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-cased",
                config=student_config,
                state_dict=distil_sd,
            )
        else:
            student_config = transformers.AutoConfig.from_pretrained(
                "distilbert-base-multilingual-cased",
                output_hidden_states=True,
                output_logits=True,
            )
            self.student = transformers.DistilBertForMaskedLM.from_pretrained(
                "distilbert-base-multilingual-cased",
                config=student_config,
                state_dict=distil_sd,
            )

    def forward(self, *model_args, **model_kwargs):
        """Forward nethod"""
        return self.student(*model_args, **model_kwargs)

    @staticmethod
    def _extract(
        teacher_model,
        layers: List[int],
        prefix_teacher: str = "bert",
        prefix_student: str = "distilbert",
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts state dict from teacher model

        Args:
            teacher_model: model to extract
            layers: layers indexes to initialize
            prefix_teacher: name of the teacher model
            prefix_student: name of the student model
        """
        state_dict = teacher_model.state_dict()
        compressed_sd = {}

        # extract embeddings
        for w in ["word_embeddings", "position_embeddings"]:
            compressed_sd[
                f"{prefix_student}.embeddings.{w}.weight"
            ] = state_dict[f"{prefix_teacher}.embeddings.{w}.weight"]
        for w in ["weight", "bias"]:
            compressed_sd[
                f"{prefix_student}.embeddings.LayerNorm.{w}"
            ] = state_dict[f"{prefix_teacher}.embeddings.LayerNorm.{w}"]
        # extract encoder
        std_idx = 0
        for teacher_idx in layers:
            for w in ["weight", "bias"]:
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.q_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.self.query.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.k_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.self.key.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.v_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.self.value.{w}"  # noqa: E501
                ]

                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.attention.out_lin.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.output.dense.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.sa_layer_norm.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.attention.output.LayerNorm.{w}"  # noqa: E501
                ]

                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.ffn.lin1.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.intermediate.dense.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.ffn.lin2.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.output.dense.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.transformer.layer.{std_idx}.output_layer_norm.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.encoder.layer.{teacher_idx}.output.LayerNorm.{w}"  # noqa: E501
                ]

            std_idx += 1
        # extract vocab
        compressed_sd[f"vocab_projector.weight"] = state_dict[
            f"cls.predictions.decoder.weight"
        ]
        compressed_sd[f"vocab_projector.bias"] = state_dict[
            f"cls.predictions.bias"
        ]

        for w in ["weight", "bias"]:
            compressed_sd[f"vocab_transform.{w}"] = state_dict[
                f"cls.predictions.transform.dense.{w}"
            ]
            compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[
                f"cls.predictions.transform.LayerNorm.{w}"
            ]

        return compressed_sd
