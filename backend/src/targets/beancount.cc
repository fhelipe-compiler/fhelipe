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

#include "include/compiled_program.h"
#include "include/extended_std.h"
#include "include/mul_cc.h"
#include "include/mul_cp.h"
#include "include/rotate_c.h"
#include "targets/gflag_utils/exe_folder_gflag_utils.h"

using namespace fhelipe;

int V = 256;
int N = 1 << 15;
int E = N / V;

DEFINE_string(filename, "", "Path to rt.df file");

int MulCPCpi(int level) {
  // Memory bandwidth bound (plaintext)
  return static_cast<int>(
      std::ceil((46.0 / 27 * level) * (2 * N) * 28 / (8 * (1ll << 10))));
}

int MulCCCpi(int level) {
  // Memory bandwidth bound (KSH)
  return 2 * MulCPCpi(level);
}

int RotateCCpi(int level) {
  // Memory bandwidth bound (KSH)
  return 2 * MulCPCpi(level);
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  std::ios_base::sync_with_stdio(false);

  auto exe_folder = ExeFolderFromFlags();
  auto filename = FLAGS_filename;

  std::unordered_map<std::string, std::vector<int>> result;
  std::vector<std::string> op_names = {MulCC::StaticTypeName(),
                                       RotateC::StaticTypeName(),
                                       MulCP::StaticTypeName()};
  int total_levels = 20;

  for (const auto& op_name : op_names) {
    result.emplace(op_name, std::vector<int>(total_levels));
  }

  auto dag = ReadFile<ct_program::CtProgram>(exe_folder / filename);
  for (const auto& node : dag.NodesInTopologicalOrder()) {
    auto level = node->Value().GetLevelInfo().Level().value();
    if (dynamic_cast<const MulCC*>(&node->Value())) {
      result[MulCC::StaticTypeName()][level]++;
    } else if (dynamic_cast<const MulCP*>(&node->Value())) {
      result[MulCP::StaticTypeName()][level]++;
    } else if (dynamic_cast<const RotateC*>(&node->Value())) {
      result[RotateC::StaticTypeName()][level]++;
    }
  }

  for (const auto& op_name : op_names) {
    for (int level : Estd::indices(total_levels)) {
      std::cout << op_name << " " << level << ": " << result[op_name][level]
                << std::endl;
    }
  }

  int cycles = 0;
  for (int level : Estd::indices(total_levels)) {
    cycles += (RotateCCpi(level) * result.at(RotateC::StaticTypeName())[level]);
    cycles += (MulCPCpi(level) * result.at(MulCP::StaticTypeName())[level]);
  }
  std::cout << "Scheduling done in " << cycles << " cycles" << std::endl;
  std::cout << "Beancounting done in " << cycles << " cycles" << std::endl;
}
