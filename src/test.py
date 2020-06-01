from catalyst import dl
import pandas as pd
import pytest  # noqa: F401
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, BertForMaskedLM, DistilBertForMaskedLM, AutoTokenizer

from .data import MLMDataset
from .runners import DistilMLMRunner
from catalyst.contrib.data.nlp import LanguageModelingDataset
from transformers.data.data_collator import DataCollatorForLanguageModeling


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
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=2)
    loaders = {"train": train_dataloader, "valid": valid_dataloader}

    model = torch.nn.ModuleDict({"teacher": teacher, "student": student})
    runner = DistilMLMRunner()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        verbose=True,
        check=True,
        callbacks={"optimizer": dl.OptimizerCallback()},
    )
    assert True


if __name__ == "__main__":
    print("test")
