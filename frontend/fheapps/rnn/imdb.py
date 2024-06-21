# $lic$
# Copyright (C) 2023-2024 by Massachusetts Institute of Technology
#
# This file is part of the Fhelipe compiler.
#
# Fhelipe is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# Fhelipe is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

import csv
import itertools
import logging
import string
import tarfile
import warnings
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Iterable, List, Optional, Tuple

import nltk
import numpy as np
import tensorfhe as tfhe
import tensorfhe.app.actions as act
import torch
import torchtext
from tensorfhe.app.actions import NnSample
from tensorfhe.app.actions.utils import download

Sample = Tuple[List[int], int]


class ImdbDataMixin:
    review_len = 200
    token_cnt = 5000

    input_shape = (review_len,)

    @classmethod
    def __stopwords(cls):
        nltk_root = act.utils.download_root() / "nltk_data"
        nltk.data.path.append(nltk_root)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logging.debug("Downloading NLTK stopwords")
            nltk.download("stopwords", quiet=True, download_dir=nltk_root)

        return nltk.corpus.stopwords.words("english")

    @classmethod
    def __download_tar(cls) -> Path:
        tar_url = (
            "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        )
        tar_path = act.utils.download_root() / "imdb" / "aclImdb_v1.tar.gz"

        if not tar_path.exists():
            logging.debug("Downloading aclImdb_v1.tar.gz")
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            download(tar_url, tar_path)

        return tar_path

    @classmethod
    def __raw_dataset(cls, split: str) -> Iterable[Tuple[str, int]]:
        tar_path = cls.__download_tar()

        with tarfile.open(tar_path) as tar:
            for m in tar:
                parts = m.name.split("/")[1:]

                if len(parts) != 3:
                    continue
                if parts[0] != split or parts[1] not in ("pos", "neg"):
                    continue
                score = int(parts[1] == "pos")

                file = tar.extractfile(m)
                if file is None:
                    continue

                text = file.read().decode()
                yield text, score

    @classmethod
    def __stemmed_dataset(cls, split) -> Iterable[Tuple[List[str], int]]:
        skipwords = {*cls.__stopwords(), *string.punctuation}
        tokenizer = torchtext.data.get_tokenizer("basic_english")
        stemmer = nltk.stem.snowball.SnowballStemmer("english")

        for review, score in cls.__raw_dataset(split):
            tokens = tokenizer(review)
            tokens = (t for t in tokens if t not in skipwords)
            tokens = (stemmer.stem(t) for t in tokens)
            tokens = list(tokens)

            yield tokens, score

    @classmethod
    @lru_cache(maxsize=1)
    def __vocab(cls) -> torchtext.vocab.Vocab:
        train_ds = cls.__stemmed_dataset("train")
        vocab = torchtext.vocab.build_vocab_from_iterator(
            (r for r, _ in train_ds),
            max_tokens=cls.token_cnt - 2,
        )
        vocab.set_default_index(cls.token_cnt - 2)
        return vocab

    @classmethod
    def __gen_dataset(cls, split: str) -> Iterable[Sample]:
        stemmed = cls.__stemmed_dataset(split)
        vocab = cls.__vocab()
        for review, score in stemmed:
            review_int = vocab.forward(review)

            review_int = review_int[: cls.review_len]
            padding = itertools.repeat(
                cls.token_cnt - 1, cls.review_len - len(review_int)
            )
            review_int = list(padding) + review_int

            yield review_int, score

    @classmethod
    def __export_dataset(cls, dataset: Iterable[Sample], path: Path) -> None:
        dataset = list(dataset)

        path.parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open(mode="w", newline="") as f:
            writer = csv.writer(f)
            for review, score in dataset:
                writer.writerow([score] + review)

    @classmethod
    def __import_dataset(cls, path: Path) -> Iterable[NnSample]:
        with Path(path).open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                score = torch.tensor(float(row[0]))
                review = [int(x) for x in row[1:]]

                yield torch.tensor(review), score

    @classmethod
    def __get_dataset(cls, split: str) -> List[NnSample]:
        path = act.utils.download_root() / "imdb" / f"{split}.csv"

        if not path.exists():
            logging.info(f"Generating IMDB {split}.csv")
            dataset = cls.__gen_dataset(split)
            cls.__export_dataset(dataset, path)
        else:
            logging.info(f"Using cached IMDB {split}.csv")

        return list(cls.__import_dataset(path))

    @classmethod
    def test_data(cls) -> List[NnSample]:
        data = cls.__get_dataset("test")

        rng = Random(422)
        rng.shuffle(data)

        return data

    @classmethod
    def train_data(cls) -> List[NnSample]:
        return cls.__get_dataset("train")
