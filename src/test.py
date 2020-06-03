from catalyst import dl
from catalyst.contrib.data.nlp import LanguageModelingDataset
import pandas as pd
import pytest  # noqa: F401
import torch
from catalyst.core import MetricAggregationCallback
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForMaskedLM,
    DistilBertForMaskedLM,
)
from .callbacks import (
    CosineLossCallback,
    KLDivLossCallback,
    MaskedLanguageModelCallback,
    MSELossCallback,
    PerplexityMetricCallback,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling

from .data import MLMDataset
from .runners import DistilMLMRunner


def test_dataset():
    """Test number of tokens"""
    dataset = MLMDataset(["Hello, world"])
    output_dict = dataset[0]
    assert output_dict["attention_mask"].sum() == 5


def test_runner():
    """Test that runner executes"""
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")
    teacher_config = AutoConfig.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, output_logits=True
    )
    teacher = BertForMaskedLM.from_pretrained(
        "bert-base-uncased", config=teacher_config
    )

    student_config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        output_hidden_states=True,
        output_logits=True,
    )
    student = DistilBertForMaskedLM.from_pretrained(
        "distilbert-base-uncased", config=student_config
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = LanguageModelingDataset(train_df["text"], tokenizer)
    valid_dataset = LanguageModelingDataset(valid_df["text"], tokenizer)

    collate_fn = DataCollatorForLanguageModeling(tokenizer).collate_batch
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=2
    )
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=collate_fn, batch_size=2
    )
    loaders = {"train": train_dataloader, "valid": valid_dataloader}

    criterions = {
        "masked_lm_loss": nn.CrossEntropyLoss(),
        "mse_loss": nn.MSELoss(),
        "cosine_loss": nn.CosineEmbeddingLoss(),
        "kl_div_loss": nn.KLDivLoss(reduction="batchmean")
    }

    callbacks = {
        "masked_lm_loss": MaskedLanguageModelCallback(),
        "mse_loss": MSELossCallback(),
        "cosine_loss": CosineLossCallback(),
        "kl_div_loss": KLDivLossCallback(),
        "loss": MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",
            metrics={
                "cosine_loss": 1.0,
                "masked_lm_loss": 1.0,
                "kl_div_loss": 1.0,
                "mse_loss": 1.0
            }
        ),
        "optimizer": dl.OptimizerCallback(),
        "perplexity": PerplexityMetricCallback()
    }

    model = torch.nn.ModuleDict({"teacher": teacher, "student": student})
    runner = DistilMLMRunner()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    runner.train(
        model=model,
        optimizer=optimizer,
        criterion=criterions,
        loaders=loaders,
        verbose=True,
        check=True,
        callbacks=callbacks,
    )
    assert True


if __name__ == "__main__":
    print("test")
