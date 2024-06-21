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

import itertools
import string
import warnings

import nltk
import numpy as np
import pandas as pd
import torch
import torchtext
from torch import nn
from torch.utils.data import DataLoader


def raw_dataset(split):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = torchtext.datasets.IMDB(split=split, root=".")

    skipwords = {*nltk.corpus.stopwords.words("english"), *string.punctuation}

    tokenizer = torchtext.data.get_tokenizer("basic_english")
    stemmer = nltk.stem.snowball.SnowballStemmer("english")

    for score, review in ds:
        score = int(score == "pos")

        review = tokenizer(review)
        review = (w for w in review if w not in skipwords)
        review = (stemmer.stem(w) for w in review)
        review = list(review)

        yield score, review


imdb_tokens = 5000
review_len = 200


def imdb_vocab():
    train_ds = raw_dataset("train")
    vocab = torchtext.vocab.release_vocab_from_iterator(
        (r for _, r in train_ds),
        max_tokens=imdb_tokens - 2,
    )
    vocab.set_default_index(imdb_tokens - 2)
    return vocab


def get_dataset(split):
    raw = raw_dataset(split)
    vocab = imdb_vocab()
    for score, review in raw:
        review = vocab.forward(review)

        review = review[:review_len]
        padding = itertools.repeat(imdb_tokens - 1, review_len - len(review))
        review = list(padding) + review

        yield np.array(review), float(score)


class ImdbClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, batch_first=False):
        super().__init__()

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.rnn(x)[1][0]
        x = self.linear(x)
        return x


class Imdb(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.embed = nn.Embedding(imdb_tokens, embedding_dim)
        self.classify = ImdbClassifier(
            input_size=embedding_dim, batch_first=True
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.classify(x)
        return torch.squeeze(x, dim=-1)


def init():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    global model
    model = Imdb().to(device)

    global opt
    opt = torch.optim.Adam(model.parameters(), weight_decay=0.0001)

    global test_data, train_data
    train_data = list(get_dataset("train"))
    test_data = list(get_dataset("test"))

    global loss_criterion
    loss_criterion = nn.BCEWithLogitsLoss()


def data_loader(data):
    dl = DataLoader(data, shuffle=True, batch_size=batch_size, pin_memory=True)

    for x, y in dl:
        yield x.to(device), y.to(device)


def prediction_err(y, target):
    prod = y * (2 * target - 1)
    return (prod <= 0).sum().item()


def train_epoch():
    model.train()

    loss, error = 0, 0

    for x, target in data_loader(train_data):
        opt.zero_grad()
        y = model(x)

        batch_loss = loss_criterion(y, target)
        batch_loss.backward()
        opt.step()

        loss += batch_loss.item()
        error += prediction_err(y, target) / len(train_data)

    train_loss.append(loss)
    train_error.append(error)

    print(
        f"Training {len(train_loss)}: loss={loss:.3f}; err={error * 100:.2f}%"
    )


def test_epoch():
    model.eval()
    error = 0

    for x, target in data_loader(test_data):
        y = model(x)
        error += prediction_err(y, target) / len(test_data)

    test_error.append(error)
    print(f"Test {len(test_error)}: err={error * 100:.2f}%")
    return error


def save_csv():
    record_types = {
        "train_loss": train_loss,
        "train_error": train_error,
        "test_error": test_error,
    }

    records = []
    for t, l in record_types.items():
        for e, v in enumerate(l):
            records.append(
                {"layers": layers, "type": t, "epoch": e, "value": v}
            )
    df = pd.DataFrame(records)
    df.to_csv(f"resnet-{layers}.csv", index=False)


def save_model(epoch):
    d = {
        "model": model.state_dict(),
    }
    torch.save(d, f"model-{epoch}-{error * 100:.0f}.pt")


if __name__ == "__main__":
    epoch_cnt = 200
    batch_size = 128

    init()
    print(f"Training on {device}")

    train_loss = []
    train_error = []
    test_error = []
    print("starting!")

    for e in range(epoch_cnt):
        train_epoch()
        error = test_epoch()
        save_model(e)

    save_csv()
