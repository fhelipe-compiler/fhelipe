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

import os
import subprocess
import tempfile
import time
from pathlib import Path

from utils import add_parent_directory, cmd, results_root

FULL_DAG_ONLY = 0;
PARTITIONS_ALLOWED = 1;
TENTACLE_BEANCOUNTING = 2;

class LattigoParameters:
    def __init__(self, bootstrapping_precision, log_scale, usable_levels):
        self.bootstrapping_precision = bootstrapping_precision
        self.log_scale = log_scale
        self.usable_levels = usable_levels

    def flags(self):
        return [
            "--bootstrapping_precision",
            self.bootstrapping_precision,
            "--log_scale",
            self.log_scale,
            "--usable_levels",
            self.usable_levels,
        ]

class Benchmark:
    def __init__(
        self,
        lattigo_parameters: LattigoParameters,
        source_code,
        root: Path,
        compiler: Path,
        scheduler: Path,
        scheduler_config: Path,
        compiler_flags,
        schedule_partitions=False,
    ):
        if isinstance(source_code, list):
            self.source_code = source_code
        else:
            self.source_code = [source_code]
        self.lattigo_parameters = lattigo_parameters
        self.root = root
        self.compiler = compiler
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.compiler_flags = compiler_flags
        self.schedule_partitions = schedule_partitions

    def name(self):
        return self.root.name

    def fhelipe_path(self):
        return self.compiler.parents[2]

    def scalar_ops(self):
        result = cmd(
            ["python"]
            + self.source_code
            + [
                "--root",
                self.root,
                "useful-ops",
            ]
        )
        return int(result)

    def lattigo_parameters(self):
        return self.lattigo_parameters


class CompiledBenchmark:
    def __init__(self, source: Benchmark, exe_folder: Path):
        self.source = source
        self.exe_folder = exe_folder

    def source_code(self):
        return self.source.source_code

    def root(self):
        return self.source.root

    def compilation_time(self):
        with open(self.exe_folder / "compile.log") as f:
            lines = f.readlines()
            compilation_time_line = list(filter(lambda line: "Compilation time (seconds):" in line, lines))[0]
            compilation_time_seconds = float(compilation_time_line.split(":")[1])
            return compilation_time_seconds

    def depth(self):
        self.draw_dag(
            add_parent_directory(
                results_root("dags") / "blah.pdf", self.name()
            ),
            "basic_parser",
            "dp_bootstrapping_pass",
        )

        with open(self.exe_folder / "basic_parser__to__dp_bootstrapping_pass") as f:
            lines = f.readlines()
            max_depth = 0
            for line in lines:
                if "depth:" in line:
                    depth = line.split("depth:" )[1].split("\\n")[0]
                    if "#" in depth:
                        continue
                    depth = int(depth)
                    if depth > max_depth:
                        max_depth = depth
            return max_depth


    def draw_dag(self, dest_path: Path, src_pass, dest_pass):
        cmd(
            [
                self.source.compiler.parents[0] / "dag_to_dot",
                "--exe_folder",
                self.exe_folder,
                "--src_pass",
                src_pass,
                "--dest_pass",
                dest_pass,
            ]
        )
        cmd(
            [
                "dot",
                "-Tpdf",
                "-o",
                dest_path,
                self.exe_folder / (src_pass + "__to__" + dest_pass),
            ]
        )

    def fhelipe_path(self):
        return self.source.fhelipe_path()

    def name(self):
        return self.source.name()

    def lattigo_parameters(self):
        return self.source.lattigo_parameters

    def bootstrapping_count(self) -> int:
        bootstrap_count = 0
        with open(self.exe_folder / "rt.df") as f:
            for line in f:
                if "BootstrapC" in line:
                    bootstrap_count += 1
        return bootstrap_count

    def total_homomorphic_operations(self):
        with open(self.exe_folder / "rt.df") as f:
            return len(f.readlines())


class CraterlakeSchedule:
    def __init__(self, source: CompiledBenchmark):
        self.source = source

    def total_time(self) -> int:
        return self.bootstrapping_time() + self.user_time()

    def lattigo_parameters(self):
        return self.source.lattigo_parameters()

    def bootstrapping_time(self) -> int:
        return (
            self.source.bootstrapping_count() * 4_000_000
        )  # 4ms per bootstrap

    def user_time(self) -> int:
        cycles = 0
        with open(self.source.exe_folder / "schedule.axel") as f:
            for line in f:
                if line.startswith("Scheduling done in"):
                    cycles += int(line.strip().split(" ")[-2])
        return cycles

    def beancounted_time(self) -> int:
        cycles = 0
        with open(self.source.exe_folder / "schedule.axel") as f:
            for line in f:
                if line.startswith("Beancounting done in"):
                    cycles += int(line.strip().split(" ")[-2])
        return cycles

    def name(self):
        return self.source.name()

def tentacle_beancount(compiled_benchmark: CompiledBenchmark) -> CraterlakeSchedule:
        schedule_content = cmd(
            [
                compiled_benchmark.source.fhelipe_path() / "backend" / "release" / "tentacle_beancount",
                "--exe_folder",
                compiled_benchmark.exe_folder,
            ]
        )
        (compiled_benchmark.exe_folder / "schedule.axel").write_bytes(
            schedule_content
        )
        return CraterlakeSchedule(compiled_benchmark)


def craterlake_schedule(
    compiled_benchmark: CompiledBenchmark,
) -> CraterlakeSchedule:
    if compiled_benchmark.source.schedule_partitions == PARTITIONS_ALLOWED:
        return partition_schedule(compiled_benchmark)
    elif compiled_benchmark.source.schedule_partitions == TENTACLE_BEANCOUNTING:
        return tentacle_beancount(compiled_benchmark)

    schedule_content = cmd(
        [
            compiled_benchmark.source.scheduler,
            compiled_benchmark.source.scheduler_config,
            compiled_benchmark.exe_folder / "rt.axel",
        ]
    )
    (compiled_benchmark.exe_folder / "schedule.axel").write_bytes(
        schedule_content
    )
    return CraterlakeSchedule(compiled_benchmark)


def nonblocking_cmd(cmd):
    print(" ".join([str(x) for x in cmd]))
    fd, path = tempfile.mkstemp()

    with os.fdopen(fd, "wb") as f:
        p = subprocess.Popen(cmd, stdout=f)

    return (p, path)


def partition_schedule(
    compiled_benchmark: CompiledBenchmark,
) -> CraterlakeSchedule:
    cmd(["rm", "-rf", compiled_benchmark.exe_folder / "schedule.axel"])
    processes = []
    for file in os.listdir(compiled_benchmark.exe_folder):
        filename = os.fsdecode(file)
        if filename.startswith("partition"):
            schedule_content = ""
            p, f = nonblocking_cmd(
                [
                    compiled_benchmark.source.scheduler,
                    compiled_benchmark.source.scheduler_config,
                    compiled_benchmark.exe_folder / filename,
                ],
            )
            processes.append((p, f, filename))

    with open(
        compiled_benchmark.exe_folder / "schedule.axel", "ab"
    ) as output_file:
        start_time = time.time()
        for (p, f, filename) in processes:
            try:
                p.wait(8 * 300 - (time.time() - start_time))  # 40min timeout
                with open(f, "rb") as fd:
                    fd.seek(0)
                    output_file.write(fd.read())
            except subprocess.TimeoutExpired:
                # Fallback
                schedule_content = cmd(
                    [
                        compiled_benchmark.source.compiler.parents[0]
                        / "beancount",
                        "--exe_folder",
                        compiled_benchmark.exe_folder,
                        "--filename",
                        "rt_" + filename,
                    ]
                )
                output_file.write(schedule_content)
    return CraterlakeSchedule(compiled_benchmark)


def read_compiled(benchmark: Benchmark) -> CompiledBenchmark:
    shared_path = cmd(
        ["python"]
        + benchmark.source_code
        + [
            "--root",
            benchmark.root,
            "init-shared",
        ]
    )

    shared_path = Path(shared_path.decode().strip())
    return CompiledBenchmark(benchmark, shared_path)


def compile(benchmark: Benchmark) -> CompiledBenchmark:
    shared_path = cmd(
        ["python"]
        + benchmark.source_code
        + [
            "--root",
            benchmark.root,
            "tdf",
        ]
    )

    shared_path = Path(shared_path.decode().strip()) / "shared"

    start_time = time.time()
    compilation_log = cmd(
        [
            benchmark.compiler,
            "--sched_dfg",
            "--exe_folder",
            shared_path,
        ]
        + benchmark.lattigo_parameters.flags()
        + benchmark.compiler_flags
    )
    with open(shared_path / "compile.log", "w") as logfile:
        logfile.write("Compilation time (seconds): " + str(time.time() - start_time) + "\n")

    return CompiledBenchmark(benchmark, shared_path)
