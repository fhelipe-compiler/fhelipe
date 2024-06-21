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

import tensorfhe.app.actions as act
from fheapps.resnet import ResNetWeights, ManualResNet, ResNet, ResNetError

from .integration_test import IntegrationTest


class TestResNet(IntegrationTest):
    make_app = ResNet

    def actions(self, layers):
        return (
            ResNetWeights(layers=layers),
            act.Cifar10TestIn(ds_equal="0001"),
            act.GenDataflow(),
            act.BackendIn(),
            act.GenOutTfhe(),
            ResNetError(),
        )

    def test_20(self):
        self._test_frontend(layers=20)

    def test_56(self):
        self._test_frontend(layers=56)


class TestResNetManual(TestResNet):
    make_app = ManualResNet
