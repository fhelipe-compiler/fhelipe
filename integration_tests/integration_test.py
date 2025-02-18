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

import shutil
import subprocess
import unittest
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import tensorfhe as tfhe
import tensorfhe.app.actions as act


def _project_root():
    return Path(__file__).resolve().parent.parent


def _str_path(path):
    try:
        wd = Path(".").resolve()
        rel_path = Path(path).relative_to(wd)
        return f"./{rel_path}"
    except ValueError:
        return str(path)


class IntegrationTest(unittest.TestCase, ABC):
    atol = 1e-4
    rtol = 1e-6
    timeout = None

    __exe_root = _project_root() / ".testing-tmp"

    @classmethod
    def setUpClass(cls):
        commands = (
            # ("scons", "deps", "--no-deps-pull"),
            ("scons", "-j32", "--release"),
        )
        for cmd in commands:
            subprocess.run(
                cmd,
                cwd=_project_root() / "backend",
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def setUp(self):
        if self.__exe_root.exists():
            shutil.rmtree(self.__exe_root)

    @abstractmethod
    def make_app(self, **kwargs):
        pass

    def gen_input(self, **kwargs):
        return (act.GenRandIn(seed=0),)

    def actions(self, **kwargs):
        return (
            *self.gen_input(**kwargs),
            act.GenDataflow(),
            act.CountUsefulOps(),
            act.GenOutTfhe(),
            act.BackendIn(),
        )

    def __backend_cmd(self, exe, path, ct_type=None):
        build_root = _project_root() / "backend" / "release"

        args = (_str_path(build_root / exe), "--exe_folder", _str_path(path))
        if ct_type is not None:
            args += ("--ct_type", ct_type)

        print("+", *args)
        subprocess.run(args, check=True, timeout=self.timeout)

    def __compare_out(self, ds):
        fe = ds.out_tfhe.import_tensors()
        be = (ds / "out_unenc").import_tensors()

        self.assertEqual(set(be.keys()), set(fe.keys()))
        for k in fe.keys():
            np.testing.assert_allclose(
                be[k],
                fe[k],
                atol=self.atol,
                rtol=self.rtol,
                err_msg=f"Mismatching outputs for {k} in {ds}",
            )

    def _test_frontend(self, **kwargs):
        app = self.make_app(**kwargs)
        exe = app.exe_manager(self.__exe_root)

        for a in self.actions(**kwargs):
            a.run(exe)

        return exe

    def _test(self, **kwargs):
        exe = self._test_frontend(**kwargs)

        if not exe.datasets:
            raise RuntimeError("No datasets")

        self.__backend_cmd("compile", exe.datasets.shared)

        for ds in exe.datasets:
            for exe in ("encrypt", "run", "decrypt"):
                self.__backend_cmd(exe, ds, ct_type="clear")

            self.__compare_out(ds)
