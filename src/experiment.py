from typing import Dict, Union
from collections import OrderedDict
from pathlib import Path

from catalyst.dl import ConfigExperiment
from catalyst.utils.tools.typing import Model, Optimizer
import pandas as pd

from .data import MLMDataset


class Experiment(ConfigExperiment):
    """
    @TODO: Docs. Contribution is welcome
    """

    def __init__(self, config: Dict):
        """
        @TODO: Docs. Contribution is welcome
        """
        super().__init__(config)
        self.config = config

    def get_transforms(self, stage: str = None, mode: str = None):
        """
        @TODO: Docs. Contribution is welcome
        """
        return []

    def get_optimizer(
        self, stage: str, model: Union[Model, Dict[str, Model]]
    ) -> Union[Optimizer, Dict[str, Optimizer]]:
        """Returns the optimizer for a given stage.
        Args:
            stage (str): stage name
            model (Union[Model, Dict[str, Model]]): model or a dict of models
        """
        optimizer_params = self.stages_config[stage].get(
            "optimizer_params", {}
        )
        key_value_flag = optimizer_params.pop("_key_value", False)

        if key_value_flag:
            optimizer = {}
            for key, params_ in optimizer_params.items():
                # load specified optimizer from checkpoint
                optimizer_key = "optimizer_key"
                assert optimizer_key not in params_, "keyword reserved"
                params_[optimizer_key] = key

                optimizer[key] = self._get_optimizer(
                    stage, model["student"], **params_
                )
        else:
            optimizer = self._get_optimizer(
                stage, model["student"], **optimizer_params
            )

        return optimizer

    # noinspection PyMethodOverriding
    def get_datasets(
        self,
        stage: str,
        path_to_data: str,
        train_filename: str,
        valid_filename: str,
        max_sequence_length: int,
        text_field: str,
        model_name: str,
        **kwargs,
    ):
        """
        @TODO: Docs. Contribution is welcome
        """
        datasets = OrderedDict()

        path_to_data = Path(path_to_data)

        train_df = pd.read_csv(path_to_data / train_filename)
        valid_df = pd.read_csv(path_to_data / valid_filename)

        train_dataset = MLMDataset(
            texts=train_df[text_field],
            max_seq_length=max_sequence_length,
            model_name=model_name,
        )

        valid_dataset = MLMDataset(
            texts=valid_df[text_field],
            max_seq_length=max_sequence_length,
            model_name=model_name,
        )

        datasets["train"] = train_dataset
        datasets["valid"] = valid_dataset

        return datasets
