/** $lic$
 * Copyright (C) 2023-2024 by Massachusetts Institute of Technology
 *
 * This file is part of the Fhelipe compiler.
 *
 * Fhelipe is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, version 3.
 *
 * Fhelipe is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>. 
 */

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <unordered_map>

#include "include/add_cc.h"
#include "include/add_cp.h"
#include "include/add_cs.h"
#include "include/bootstrap_c.h"
#include "include/compiled_program.h"
#include "include/constants.h"
#include "include/ct_program.h"
#include "include/filesystem_utils.h"
#include "include/input_c.h"
#include "include/mul_cc.h"
#include "include/mul_cp.h"
#include "include/mul_cs.h"
#include "include/output_c.h"
#include "include/rotate_c.h"
#include "include/t_bootstrap_c.h"
#include "include/t_cyclic_shift_c.h"
#include "include/t_layout_conversion_c.h"
#include "include/tensor_layout.h"
#include "targets/gflag_utils/exe_folder_gflag_utils.h"

using namespace fhelipe;

namespace {

DEFINE_bool(full, false, "Set flag to see runtime breakdown per TOp.");

double chunk_size_modular_multiplies_per_second = (1 << (30 - 16));

bool RequiresKeyswitching(const CtOp& ct_op) {
  return dynamic_cast<const MulCC*>(&ct_op) ||
         dynamic_cast<const RotateC*>(&ct_op);
}

double ExecutionTime(const CtOp& ct_op) {
  // 1 time unit corresponds to 2^16 CPU modular multiply, roughly
  if (const auto* input_c = dynamic_cast<const InputC*>(&ct_op)) {
    return 0;
  }
  if (const auto* output_c = dynamic_cast<const OutputC*>(&ct_op)) {
    return 0;
  }
  if (RequiresKeyswitching(ct_op)) {
    // Magic numbers from CraterLake Table 1
    return static_cast<double>(3 * ct_op.GetLevel().value() *
                                   ct_op.GetLevel().value() +
                               (4 + 6 * 8) * ct_op.GetLevel().value()) /
           chunk_size_modular_multiplies_per_second;
  }
  if (dynamic_cast<const BootstrapC*>(&ct_op)) {
    return 17;  // 17 seconds per bootstrap
  }
  return ct_op.GetLevel().value() / chunk_size_modular_multiplies_per_second;
}

class Perf {
 public:
  Perf() = default;

  void RecordTime(const TOp& t_op, const double& time) {
    total_time_ += time;
    if (dynamic_cast<const TBootstrapC*>(&t_op)) {
      time_breakdown_["TBootstrapC"] += time;
    } else if (dynamic_cast<const TLayoutConversionC*>(&t_op)) {
      time_breakdown_["TLayoutConversionC"] += time;
    } else if (dynamic_cast<const TMulCC*>(&t_op)) {
      time_breakdown_["TMulCC"] += time;
    } else if (dynamic_cast<const TMulCP*>(&t_op)) {
      time_breakdown_["TMulCP"] += time;
    } else if (dynamic_cast<const TMulCSI*>(&t_op)) {
      time_breakdown_["TMulCSI"] += time;
    } else if (dynamic_cast<const TAddCC*>(&t_op)) {
      time_breakdown_["TAddCC"] += time;
    } else if (dynamic_cast<const TAddCP*>(&t_op)) {
      time_breakdown_["TAddCP"] += time;
    } else if (dynamic_cast<const TAddCSI*>(&t_op)) {
      time_breakdown_["TAddCSI"] += time;
    } else if (dynamic_cast<const TUnpaddedShiftC*>(&t_op)) {
      time_breakdown_["TUnpaddedShiftC"] += time;
    } else if (dynamic_cast<const TCyclicShiftC*>(&t_op)) {
      time_breakdown_["TCyclicShiftC"] += time;
    }
  }

  const std::unordered_map<std::string, double> Breakdown() const {
    return time_breakdown_;
  }

  double TotalTime() const { return total_time_; }

 private:
  std::unordered_map<std::string, double> time_breakdown_;
  double total_time_;
};

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::ios_base::sync_with_stdio(false);
  auto exe_folder = ExeFolderFromFlags();
  if (FLAGS_full) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    auto debug_info =
        compiled_program.GetDebugInfo<LevelingPassOutput, CtOpOptimizerOutput,
                                      LeveledTOp, CtOp>(
            compiled_program.LastLevelingPassName(),
            compiled_program.LastPassName());
    Perf perf;
    for (const auto& [src, dest] : debug_info.Mappings()) {
      CHECK(src.size() <= 1);
      if (src.empty()) {
        continue;
      }
      auto times = Estd::transform(dest, [](const auto* ct_op) {
        return ExecutionTime(ct_op->Value());
      });
      perf.RecordTime(src.at(0)->Value().GetTOp(), Estd::sum(times));
    }
    std::cout << "Total time: "
              << perf.TotalTime() / chunk_size_modular_multiplies_per_second
              << "s" << std::endl;
    for (const auto& [type, time] : perf.Breakdown()) {
      std::cout << type << ": " << time << " (" << 100 * time / perf.TotalTime()
                << "%)" << std::endl;
    }
  } else {
    const auto& dag = ReadFile<ct_program::CtProgram>(exe_folder / kExecutable);

    double total_time = 0;
    double total_bootstrapping_time = 0;
    for (const auto& node : dag.NodesInTopologicalOrder()) {
      total_time += ExecutionTime(node->Value());
      if (dynamic_cast<const BootstrapC*>(&node->Value())) {
        total_bootstrapping_time += ExecutionTime(node->Value());
      }
    }
    std::cout << "Total time: " << total_time << "s" << std::endl;
    std::cout << "Total bootstrapping time: " << total_bootstrapping_time
              << "s ("
              << static_cast<int>(100 * total_bootstrapping_time / total_time)
              << "%)" << std::endl;
  }
}
