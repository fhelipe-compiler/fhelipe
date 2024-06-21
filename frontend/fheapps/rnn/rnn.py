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

from pathlib import Path
from typing import overload

import numpy as np
import tensorfhe as tfhe
import tensorfhe.app.actions as act
import tensorfhe.nn as nn
import torch
from fheapps.rnn.imdb import ImdbDataMixin
from tensorfhe import Input, Tensor, TorchInput, _result_repack
from tensorfhe.nn import ModuleV


class Embedding(torch.nn.Module):
    def __init__(self, token_cnt: int, embedding_size: int):
        super().__init__()
        self.embed = torch.nn.Embedding(token_cnt, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x.int())


@_result_repack
def approx_tanh(x: tfhe.VectorT) -> tfhe.VectorT:
    c_1 = 0.249476365628036
    c_3 = -0.00163574303018748

    return c_1 * x + (c_3 * x) * (x * x)


class RNN(nn.module.WeakSplitModule):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.__in_size = input_size
        self.__h_size = hidden_size

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[1][0]

    def forward_fhe(self, x: Input) -> Tensor:
        w_ih = self.weights("rnn.weight_ih_l0", (self.__h_size, self.__in_size))
        w_hh = self.weights("rnn.weight_hh_l0", (self.__h_size, self.__h_size))
        b = self.weights("rnn.bias_ih_l0", (self.__h_size,)) + self.weights(
            "rnn.bias_hh_l0", (self.__h_size,)
        )

        h: Tensor = tfhe.as_input(0).broadcast_to((self.__h_size,))

        for i in range(x.shape[0]):
            x_enc = x[i].enc()
            pre_act = (
                tfhe.lib.mul_mv(w_ih, x_enc) + tfhe.lib.mul_mv(w_hh, h) + b
            )
            h = approx_tanh(pre_act)

            # for usable_levels == 10
            if i % 2 == 1:
                h = h._bootstrap()
        return h


class ImdbRNN(nn.NN):
    input_shape = ImdbDataMixin.input_shape

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.embed = nn.Preprocess(
            Embedding(ImdbDataMixin.token_cnt, input_size)
        )
        self.rnn = RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    @overload
    def forward(self, x: TorchInput) -> TorchInput:
        ...

    @overload
    def forward(self, x: Input) -> Tensor:
        ...

    def forward(self, x: Input) -> Tensor:
        x = self.embed(x)
        x_t = self.rnn(x)
        x_t = self.linear(x_t)
        x_t = x_t.drop_dim(dim=-1)
        return x_t


class ImdbTestIn(ImdbDataMixin, act.GenTestIn):
    pass


class ImdbTestError(act.BinaryOutputMixin, ImdbDataMixin, act.TestError):
    pass


class RnnTrain(act.BinaryOutputMixin, ImdbDataMixin, act.TrainNn):
    epoch_cnt = 100
    batch_size = 128

    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), weight_decay=0.0001)

    def loss_f(self) -> act.LossT:
        return torch.nn.BCEWithLogitsLoss()


class RnnWeights(act.PopulateSharedCheckpoint):
    checkpoint = Path(__file__).parent / "checkpoint.pt"


class RnnApp(tfhe.NnApp):
    actions = {
        *act.common_nn_actions,
        ImdbTestIn,
        ImdbTestError,
        RnnTrain,
        RnnWeights,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(nn=ImdbRNN(128, 128))


if __name__ == "__main__":
    RnnApp().main()
