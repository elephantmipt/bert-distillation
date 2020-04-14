import pytest  # noqa: F401

import torch  # noqa: F401

from .data import MLMDataset


def test_dataset():
    dataset = MLMDataset(["Hello, world"])
    output_dict = dataset[0]
    assert output_dict["attention_mask"].sum() == 5


if __name__ == "__main__":
    print("test")
