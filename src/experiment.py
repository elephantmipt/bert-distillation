from typing import Dict, Union
from collections import OrderedDict
import logging
from pathlib import Path

from catalyst.dl import ConfigExperiment, utils
from catalyst.tools.typing import Model, Optimizer
import pandas as pd
from src.data import LanguageModelingDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


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

        data_params = dict(self.stages_config[stage]["data_params"])
        model_name = data_params["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = LanguageModelingDataset(
            texts=train_df[text_field],
            max_seq_length=max_sequence_length,
            tokenizer=tokenizer,
            sort=False,
            lazy=True,
        )

        valid_dataset = LanguageModelingDataset(
            texts=valid_df[text_field],
            max_seq_length=max_sequence_length,
            tokenizer=tokenizer,
            sort=False,
            lazy=True,
        )

        datasets["train"] = train_dataset
        datasets["valid"] = valid_dataset

        return datasets

    def get_loaders(
        self, stage: str, epoch: int = None,
    ) -> "OrderedDict[str, DataLoader]":
        """
        Returns loaders for the stage
        Args:
            stage: string with stage name
            epoch: epoch

        Returns:
            Dict of loaders
        """
        data_params = dict(self.stages_config[stage]["data_params"])
        model_name = data_params["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        collate_fn = DataCollatorForLanguageModeling(tokenizer).collate_batch
        loaders_params = {
            "train": {"collate_fn": collate_fn},
            "valid": {"collate_fn": collate_fn},
        }
        loaders = utils.get_loaders_from_params(
            get_datasets_fn=self.get_datasets,
            initial_seed=self.initial_seed,
            stage=stage,
            loaders_params=loaders_params,
            **data_params,
        )

        return loaders
