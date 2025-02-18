#$lic$
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

import copy
import getpass
from pathlib import Path

import flags
from benchmark import (
    Benchmark,
    CompiledBenchmark,
    CraterlakeSchedule,
    LattigoParameters,
    compile,
    craterlake_schedule,
    partition_schedule,
    read_compiled,
    TENTACLE_BEANCOUNTING,
)
from utils import cmd

default_bootstrapping_precision = 19
default_log_scale = 45
default_usable_levels = 10
squeezenet_log_scale = 35
squeezenet_usable_levels = 10

compiler = flags.fhelipe_path / "backend/release/compile"
scheduler = Path(
    "external/craterlake_scheduler/scheduler/build/schedule"
)
scheduler_config = Path(
    "external/craterlake_scheduler/scheduler/cfgs/default_65536.cfg"
)

runnable_compiler_flags = []
default_compiler_flags = ["--avoid_writes"]
default_lattigo_parameters = LattigoParameters(
    default_bootstrapping_precision, default_log_scale, default_usable_levels
)
squeezenet_lattigo_parameters = LattigoParameters(
    default_bootstrapping_precision, squeezenet_log_scale, squeezenet_usable_levels)
condor_squeezenet_lattigo_parameters = LattigoParameters(
    default_bootstrapping_precision, squeezenet_log_scale, squeezenet_usable_levels)

logreg_log_scale = 35
default_logreg_parameters = LattigoParameters(
    default_bootstrapping_precision, logreg_log_scale, default_usable_levels
)

default_manual_flags = default_compiler_flags + ["--leveling_pass", "noop"]
default_chet_flags = [
    "--layout_pass",
    "chet",
    "--leveling_pass",
    "chet_lazy",
]

lola_mnist_lattigo_parameters = LattigoParameters(
    default_bootstrapping_precision, 35, 16
)

ttm_mttkrp_lattigo_parameters = LattigoParameters(
    default_bootstrapping_precision, default_log_scale, 11
)


fhelipe_lola_mnist = Benchmark(
    lola_mnist_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/lola/lola_mnist.py"),
    flags.folder / "fhelipe_lola_mnist",
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

manual_lola_mnist = Benchmark(
    lola_mnist_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/lola/manual_mnist.py"),
    flags.folder / "manual_lola_mnist",
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

chet_lola_mnist = Benchmark(
    lola_mnist_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/lola/lola_mnist.py"),
    flags.folder / "chet_lola_mnist",
    compiler,
    scheduler,
    scheduler_config,
    default_chet_flags,
)


fhelipe_resnet = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/resnet/resnet.py"),
    flags.folder / Path("fhelipe_resnet"),
    compiler,
    scheduler,
    scheduler_config,
    default_compiler_flags,
)

fhelipe_aespa_resnet = Benchmark(
    default_lattigo_parameters,
    [flags.fhelipe_path / Path("frontend/fheapps/nn_resnet/resnet.py"), "+a", "herpn"],
    flags.folder / Path("fhelipe_aespa_resnet"),
    compiler,
    scheduler,
    scheduler_config,
    default_compiler_flags,
)

fhelipe_squeezenet = Benchmark(
        squeezenet_lattigo_parameters,
        flags.fhelipe_path / Path("frontend/fheapps/squeezenet/squeezenet.py"),
        flags.folder / Path("fhelipe_squeezenet"),
        compiler,
        scheduler,
        scheduler_config,
        default_compiler_flags)

chet_squeezenet = Benchmark(
        squeezenet_lattigo_parameters,
        flags.fhelipe_path / Path("frontend/fheapps/squeezenet/squeezenet.py"),
        flags.folder / Path("chet_squeezenet"),
        compiler,
        scheduler,
        scheduler_config,
        default_chet_flags,
        schedule_partitions=True,
)

# RNS-CKKS
condor_fhelipe_resnet = Benchmark(
    LattigoParameters(
        default_bootstrapping_precision, 35, default_usable_levels
    ),
    flags.fhelipe_path / Path("frontend/fheapps/resnet/resnet.py"),
    flags.folder / "condor_fhelipe_resnet",
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

condor_fhelipe_aespa_resnet = Benchmark(
    default_lattigo_parameters,
    [flags.fhelipe_path / Path("frontend/fheapps/nn_resnet/resnet.py"), "+a", "herpn"],
    flags.folder / Path("condor_fhelipe_aespa_resnet"),
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

condor_fhelipe_rnn = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/rnn/rnn.py"),
    flags.folder / "condor_fhelipe_rnn",
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

condor_fhelipe_squeezenet = Benchmark(
        condor_squeezenet_lattigo_parameters,
        flags.fhelipe_path / Path("frontend/fheapps/squeezenet/squeezenet.py"),
        flags.folder / "condor_fhelipe_squeezenet",
        compiler,
        scheduler,
        scheduler_config,
        runnable_compiler_flags)

condor_fhelipe_logreg = Benchmark(
    default_logreg_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/logreg/logreg.py"),
    flags.folder / Path("condor_fhelipe_logreg"),
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

manual_resnet = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/resnet/manual.py"),
    flags.folder / Path("manual_resnet"),
    compiler,
    scheduler,
    scheduler_config,
    default_manual_flags,
)

chet_resnet = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/resnet/resnet.py"),
    flags.folder / "chet_resnet",
    compiler,
    scheduler,
    scheduler_config,
    default_chet_flags,
    schedule_partitions=True,
)

fhelipe_logreg = Benchmark(
    default_logreg_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/logreg/logreg.py"),
    flags.folder / Path("fhelipe_logreg"),
    compiler,
    scheduler,
    scheduler_config,
    default_compiler_flags,
)

manual_logreg = Benchmark(
    default_logreg_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/logreg/manual.py"),
    flags.folder / Path("manual_logreg"),
    compiler,
    scheduler,
    scheduler_config,
    default_manual_flags,
)

chet_logreg = Benchmark(
    default_logreg_parameters,
    [flags.fhelipe_path / Path("frontend/fheapps/logreg/logreg.py"), "+chet"],
    flags.folder / Path("chet_logreg"),
    compiler,
    scheduler,
    scheduler_config,
    default_chet_flags,
    schedule_partitions=True,
)

fhelipe_rnn = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/rnn/rnn.py"),
    flags.folder / Path("fhelipe_rnn"),
    compiler,
    scheduler,
    scheduler_config,
    default_compiler_flags,
)

manual_rnn = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/rnn/rnn.py"),
    flags.folder / Path("manual_rnn"),
    compiler,
    scheduler,
    scheduler_config,
    default_manual_flags,
)

chet_rnn = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/rnn/rnn.py"),
    flags.folder / Path("chet_rnn"),
    compiler,
    scheduler,
    scheduler_config,
    default_chet_flags,
    schedule_partitions=True,
)

fhelipe_fft = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/tensor/fft.py"),
    flags.folder / Path("fhelipe_fft"),
    compiler,
    scheduler,
    scheduler_config,
    default_compiler_flags,
)

fhelipe_ttm = Benchmark(
    ttm_mttkrp_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/tensor/ttm.py"),
    flags.folder / Path("fhelipe_ttm"),
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

fhelipe_mttkrp = Benchmark(
    ttm_mttkrp_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/tensor/mttkrp.py"),
    flags.folder / Path("fhelipe_mttkrp"),
    compiler,
    scheduler,
    scheduler_config,
    runnable_compiler_flags,
)

chet_fft = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/tensor/fft.py"),
    flags.folder / Path("chet_fft"),
    compiler,
    scheduler,
    scheduler_config,
    default_chet_flags,
    schedule_partitions=True,
)

chet_ttm = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/tensor/ttm.py"),
    flags.folder / Path("chet_ttm"),
    compiler,
    scheduler,
    scheduler_config,
    default_chet_flags + ["--ct_op_pass", "dummy"],
    schedule_partitions=TENTACLE_BEANCOUNTING,
)

chet_mttkrp = Benchmark(
    default_lattigo_parameters,
    flags.fhelipe_path / Path("frontend/fheapps/tensor/mttkrp.py"),
    flags.folder / Path("chet_mttkrp"),
    compiler,
    scheduler,
    scheduler_config,
    default_chet_flags + ["--ct_op_pass", "dummy"],
    schedule_partitions=TENTACLE_BEANCOUNTING,
)

fhelipe_ml_benchmarks = [fhelipe_logreg, fhelipe_resnet,
        fhelipe_rnn, fhelipe_squeezenet]
fhelipe_tensor_benchmarks = [fhelipe_fft, fhelipe_ttm, fhelipe_mttkrp]
fhelipe_benchmarks = fhelipe_ml_benchmarks + fhelipe_tensor_benchmarks
manual_benchmarks = [manual_resnet, manual_logreg, manual_rnn, manual_lola_mnist]
chet_benchmarks = [chet_squeezenet, chet_resnet, chet_logreg, chet_rnn,
        chet_fft, chet_ttm, chet_mttkrp, chet_lola_mnist]

benchmarks = fhelipe_benchmarks + manual_benchmarks + chet_benchmarks

cpu_benchmarks = []
for benchmark in fhelipe_benchmarks + manual_benchmarks:
    cpu_benchmark = copy.deepcopy(benchmark)
    cpu_benchmark.compiler_flags = list(
        filter(lambda x: x != "--avoid_writes", cpu_benchmark.compiler_flags)
    )
    cpu_benchmarks.append(cpu_benchmark)

benchmark_dict = {
    "fhelipe_lola_mnist": fhelipe_lola_mnist,
    "manual_lola_mnist": manual_lola_mnist,
    "chet_lola_mnist": chet_lola_mnist,
    "fhelipe_resnet": fhelipe_resnet,
    "fhelipe_squeezenet": fhelipe_squeezenet,
    "fhelipe_logreg": fhelipe_logreg,
    "fhelipe_rnn": fhelipe_rnn,
    "condor_fhelipe_resnet": condor_fhelipe_resnet,
    "condor_fhelipe_squeezenet": condor_fhelipe_squeezenet,
    "condor_fhelipe_logreg": condor_fhelipe_logreg,
    "condor_fhelipe_rnn": condor_fhelipe_rnn,
    "fhelipe_fft": fhelipe_fft,
    "fhelipe_ttm": fhelipe_ttm,
    "fhelipe_mttkrp": fhelipe_mttkrp,
    "chet_squeezenet": chet_squeezenet,
    "chet_logreg": chet_logreg,
    "chet_resnet": chet_resnet,
    "chet_rnn": chet_rnn,
    "chet_fft": chet_fft,
    "chet_ttm": chet_ttm,
    "chet_mttkrp": chet_mttkrp,
    "fhelipe_aespa_resnet": fhelipe_aespa_resnet,
    "condor_fhelipe_aespa_resnet": condor_fhelipe_aespa_resnet,
    "manual_resnet": manual_resnet,
    "manual_rnn": manual_rnn,
    "manual_logreg": manual_logreg,
}
