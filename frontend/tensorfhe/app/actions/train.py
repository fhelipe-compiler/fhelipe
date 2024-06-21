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

import logging
import time
from abc import abstractmethod
from argparse import ArgumentParser
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Protocol,
    Tuple,
)

import torch
from torch.utils.data import DataLoader

from ...core import TensorV
from ..datapath import DataPath
from ..exe_manager import NnExeManager
from .action import NnAction
from .utils import AvgStats, NnData, NnSample, prediction_error

LossT = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Scheduler(Protocol):
    def step(self) -> None:
        ...

    def load_state_dict(self, state) -> None:
        ...

    def state_dict(self) -> Any:
        ...


class Checkpoint:
    __slots__ = (
        "train_loss",
        "train_error",
        "test_error",
        "model",
        "best_model",
        "opt",
        "sch",
    )

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.train_error: List[float] = []
        self.test_error: List[float] = []

        self.model: Mapping[str, TensorV] = {}
        self.best_model: Mapping[str, TensorV] = {}
        self.opt: Any = {}
        self.sch: Any = {}

    @property
    def epoch(self) -> int:
        return len(self.train_loss)

    def epoch_str(self, epoch: Optional[int] = None) -> str:
        if epoch is None:
            epoch = self.epoch - 1
        return (
            f"{epoch}: "
            f"test_err={100 * self.test_error[epoch]:.2f}%; "
            f"train_err={100 * self.train_error[epoch]:.2f}%; "
            f"train_loss={self.train_loss[epoch]:.3f};"
        )

    @classmethod
    def load(cls, path: DataPath) -> "Checkpoint":
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint doesn't exist!", path)

        logging.info(f"Loading checkpoint from {path}...")
        d = torch.load(Path(path), map_location=torch.device("cpu"))

        cp = Checkpoint()
        for s in cls.__slots__:
            setattr(cp, s, d[s])

        logging.info("Last epoch " + cp.epoch_str(cp.epoch - 1))
        logging.info("Best epoch " + cp.epoch_str(cp.min_error()[1]))

        return cp

    def save(self, path: DataPath) -> None:
        logging.info(f"Saving checkpoint to {path} ({self.epoch} epochs).")
        d = {s: getattr(self, s) for s in self.__slots__}
        path.parent.mkdir()
        torch.save(d, Path(path))

    def min_error(self) -> Tuple[float, int]:
        return min((err, epoch) for epoch, err in enumerate(self.test_error))


class TrainNn(NnAction):
    name = "train"
    help = "Train model in PyTorch"

    epoch_cnt: int = 100
    batch_size: int = 128
    checkpoint_epochs: int = 5

    @abstractmethod
    def train_data(self) -> NnData:
        raise NotImplementedError

    @abstractmethod
    def test_data(self) -> NnData:
        raise NotImplementedError

    @abstractmethod
    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        raise NotImplementedError

    @abstractmethod
    def loss_f(self) -> LossT:
        raise NotImplementedError

    def scheduler(self, optimizer: torch.optim.Optimizer) -> Scheduler:
        # Constant learning rate
        return torch.optim.lr_scheduler.ConstantLR(optimizer, 1)

    def output_transform(self, x: TensorV) -> TensorV:
        return x

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "--no-cuda",
            action="store_false",
            dest="use_cuda",
            help="Run on CPU, even if a GPU is available",
        )

    def __init__(self, use_cuda: bool = True, **kwargs):
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device: Final = (
            torch.device("cuda") if use_cuda else torch.device("cpu")
        )

    @cached_property
    def _train_data(self) -> NnData:
        return self.train_data()

    @cached_property
    def _test_data(self) -> NnData:
        return self.test_data()

    @cached_property
    def _loss_f(self) -> LossT:
        return self.loss_f()

    def __init_logging(self, path: DataPath) -> None:
        path.parent.mkdir()
        logging.getLogger().addHandler(logging.FileHandler(path))

    def run(self, exe: NnExeManager) -> float:
        self.__init_logging(exe.root.training.log)
        checkpoint_path = exe.root.training.checkpoint

        logging.info(f"Training device: {self.device}")

        exe.nn.torch.to(self.device)
        state = TrainState(self, exe.nn.torch)

        try:
            state.load_checkpoint(Checkpoint.load(checkpoint_path))
        except FileNotFoundError:
            logging.info("Checkpoint missing; starting training from scratch.")

        while state.cp.epoch < self.epoch_cnt:
            start_t = time.perf_counter()

            state.training_epoch()
            state.test_epoch()

            end_t = time.perf_counter()
            logging.info(
                state.cp.epoch_str() + f" duration: {end_t - start_t:.2f}s"
            )

            if (
                state.cp.epoch % self.checkpoint_epochs == 0
                or state.cp.epoch == self.epoch_cnt
            ):
                state.update_checkpoint()
                state.cp.save(checkpoint_path)

        min_error = state.cp.min_error()[0]
        logging.info(f"Min error: {100 * min_error:.2f}")
        return min_error


class TrainState:
    cp: Checkpoint

    def __init__(self, action: TrainNn, model: torch.nn.Module):
        self.__a: Final = action

        self.model: Final = model
        self.opt: Final = action.optimizer(self.model)
        self.sch: Final = action.scheduler(self.opt)

        self.cp = Checkpoint()

    def load_checkpoint(self, cp: Checkpoint) -> None:
        self.cp = cp

        self.model.load_state_dict(OrderedDict(cp.model))
        self.opt.load_state_dict(cp.opt)
        self.sch.load_state_dict(cp.sch)

    def update_checkpoint(self) -> None:
        self.cp.model = self.model.state_dict()
        self.cp.opt = self.opt.state_dict()
        self.cp.sch = self.sch.state_dict()

    def __data_loader(
        self, data: NnData, shuffle: bool = False
    ) -> Iterator[NnSample]:
        dl: DataLoader = DataLoader(
            data,  # type: ignore[arg-type]
            batch_size=self.__a.batch_size,
            pin_memory=True,
            shuffle=shuffle,
        )

        for x, y in dl:
            yield x.to(self.__a.device), y.to(self.__a.device)

    def training_epoch(self) -> None:
        loss: float = 0
        err_stats = AvgStats()

        data_loader = self.__data_loader(self.__a._train_data, shuffle=True)

        self.model.eval()  # Reset ReLU's running maximums
        self.model.train()

        for x, target in data_loader:
            self.opt.zero_grad()
            y = self.model(x)

            batch_loss = self.__a._loss_f(y, target)
            batch_loss.backward()
            self.opt.step()

            loss += batch_loss.item()

            y = self.__a.output_transform(y.detach())
            err_stats += prediction_error(y, target)

        self.sch.step()

        error = float(err_stats)
        self.cp.train_loss.append(loss)
        self.cp.train_error.append(error)

    def test_epoch(self) -> None:
        data_loader = self.__data_loader(self.__a._test_data)
        err_stats = AvgStats()

        self.model.eval()

        with torch.no_grad():
            for x, target in data_loader:
                y = self.model(x)

                y = self.__a.output_transform(y.detach())
                err_stats += prediction_error(y, target)

        error = float(err_stats)
        self.cp.test_error.append(error)

        if self.cp.min_error()[1] == self.cp.epoch - 1:
            logging.info(f"Min error so far!")
            self.cp.best_model = deepcopy(self.model.state_dict())
