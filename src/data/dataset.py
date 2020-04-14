from collections import Counter
import logging
import math
from typing import List, Mapping

from tqdm.auto import tqdm
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    """Wrapper around Torch Dataset to perform text classification."""

    def __init__(
        self,
        texts: List[str],
        max_seq_length: int = 512,
        model_name: str = "distilbert-base-uncased",
        probs_smothing: float = .75,
        mask_prob: float = .5,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            texts (List[str]): a list with texts to classify or to train the
                classifier on
            max_seq_length (int): maximal sequence length in tokens,
                texts will be stripped to this length
            model_name (str): transformer model name, needed to perform
                appropriate tokenization
            probs_smothing (float): power of token probabilities
            mask_prob (float): probability of mask appearance
        """
        self.texts = texts
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # suppresses tokenizer warnings
        logging.getLogger("transformers.tokenization_utils").setLevel(
            logging.FATAL
        )

        # special tokens for transformers
        # in the simplest case a [CLS] token is added in the beginning
        # and [SEP] token is added in the end of a piece of text
        # [CLS] <indexes text tokens> [SEP] .. <[PAD]>
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]
        self.mask_vid = self.tokenizer.vocab["[MASK]"]
        self.mask_prob = mask_prob
        self.device = device

        word_counter = Counter()
        pbar = tqdm(texts,
                    desc="Counting probabilities")
        for text in pbar:
            text_encoded = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).squeeze(0)
            word_counter.update(text_encoded)

        counts = torch.zeros(self.tokenizer.vocab_size)

        self.vocab_size = self.tokenizer.vocab_size

        for k, v in word_counter.items():
            counts[k] = v
        self.counts = counts
        self.token_probs =\
            (self.counts / self.counts.sum()) ** probs_smothing

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        # encoding the text
        x = self.texts[index]
        x_encoded = self.tokenizer.encode(
            x,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).squeeze(0)

        # padding short texts
        true_seq_length = x_encoded.size(0)
        pad_size = self.max_seq_length - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        x_tensor = torch.cat((x_encoded, pad_ids))

        x_prob = self.token_probs[x_tensor.flatten()]
        mask_number = math.ceil(self.mask_prob * true_seq_length)

        tgt_ids = torch.multinomial(x_prob / x_prob.sum(),
                                    mask_number, replacement=False)

        mlm_labels = x_tensor.new(x_tensor.size()).copy_(x_tensor)
        pred_mask = torch.zeros(
            self.max_seq_length,
            dtype=torch.bool,
            device=self.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility

        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(self.max_seq_length)

        pred_mask[x_tensor == self.pad_vid] = 0

        x_tensor[pred_mask] = self.mask_vid

        mlm_labels[~pred_mask] = -100
        # previously `mlm_labels[1-pred_mask] = -1`,
        # cf pytorch 1.2.0 compatibility

        attention_mask = torch.ones_like(x_encoded, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        attention_mask = torch.cat((attention_mask, mask_pad)).type(torch.bool)
        output_dict = {
            "features": x_tensor,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
        }
        return output_dict
