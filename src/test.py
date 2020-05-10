from catalyst import dl
import pandas as pd
import pytest  # noqa: F401
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, BertForMaskedLM, DistilBertForMaskedLM

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
    train_dataset = MLMDataset(train_df["text"])
    valid_dataset = MLMDataset(valid_df["text"])

    train_dataloader = DataLoader(train_dataset, batch_size=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2)
    loaders = {"train": train_dataloader, "valid": valid_dataloader}

    model = torch.nn.ModuleDict({"teacher": teacher, "student": student})
    runner = DistilMLMRunner()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        verbose=True,
        num_epochs=3,
        callbacks={"optimizer": dl.OptimizerCallback()},
    )
    assert True


if __name__ == "__main__":
    print("test")
