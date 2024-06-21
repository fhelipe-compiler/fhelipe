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

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

#include "include/compiled_program.h"
#include "include/constants.h"
#include "include/dag.h"
#include "include/dag_dot.h"
#include "include/dag_io.h"
#include "include/debug_info.h"
#include "include/laid_out_tensor.h"
#include "include/leveled_t_op.h"
#include "include/pass_utils.h"
#include "include/plaintext_chunk.h"
#include "include/scaled_t_op.h"
#include "include/t_op.h"
#include "include/t_op_embrio.h"
#include "targets/gflag_utils/exe_folder_gflag_utils.h"

using namespace fhelipe;

namespace {

std::vector<PassName> parsers = {PassName{"basic_parser"}};
std::vector<PassName> layout_optimizers = {
    PassName{"fill_gaps_layout_pass"}, PassName{"conversion_decomposer_pass"},
    PassName{"layout_hoisting_pass"},  PassName{"value_numbering_pass"},
    PassName{"input_layout_pass"},     PassName{"chet_layout_pass"},
    PassName{"merge_mul_chains_pass"}};
std::vector<PassName> rescalers = {PassName{"waterline_rescale"}};
std::vector<PassName> leveling_optimizers = {
    PassName{"dp_bootstrapping_pass"}, PassName{"lazy_bootstrapping_pass"},
    PassName{"noop_leveling_pass"}, PassName{"bootstrap_prunning_pass"},
    PassName{"lazy_bootstrapping_on_chet_repacks_pass"}};
std::vector<PassName> ct_op_optimizers = {PassName{"dummy_ct_op_pass"},
                                          PassName{"basic_ct_op_pass"},
                                          PassName{"level_minimization_pass"}};

DEFINE_string(src_pass, "basic_parser", "Source pass name");
DEFINE_string(dest_pass, "", "Destination pass name");

PassName SourcePassFromFlags() { return PassName{FLAGS_src_pass}; }

PassName DestinationPassFromFlags() { return PassName{FLAGS_dest_pass}; }

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  std::ios_base::sync_with_stdio(false);

  auto exe_folder = ExeFolderFromFlags();
  auto source_pass = SourcePassFromFlags();
  auto destination_pass = DestinationPassFromFlags();

  auto stream = std::ofstream(exe_folder / (ToString(source_pass) + "__to__" +
                                            ToString(destination_pass)));

  if (Estd::contains(parsers, source_pass) &&
      Estd::contains(parsers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<ParserOutput, ParserOutput, TOpEmbrio, TOpEmbrio>(
        stream,
        compiled_program
            .GetDebugInfo<ParserOutput, ParserOutput, TOpEmbrio, TOpEmbrio>(
                source_pass, destination_pass));
  } else if (Estd::contains(parsers, source_pass) &&
             Estd::contains(layout_optimizers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<ParserOutput, LayoutPassOutput, TOpEmbrio, TOp>(
        stream,
        compiled_program
            .GetDebugInfo<ParserOutput, LayoutPassOutput, TOpEmbrio, TOp>(
                source_pass, destination_pass));
  } else if (Estd::contains(parsers, source_pass) &&
             Estd::contains(rescalers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<ParserOutput, RescalingPassOutput, TOpEmbrio,
                           ScaledTOp>(
        stream, compiled_program.GetDebugInfo<ParserOutput, RescalingPassOutput,
                                              TOpEmbrio, ScaledTOp>(
                    source_pass, destination_pass));
  } else if (Estd::contains(rescalers, source_pass) &&
             Estd::contains(leveling_optimizers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<RescalingPassOutput, LevelingOptimizerOutput,
                           ScaledTOp, LeveledTOp>(
        stream, compiled_program
                    .GetDebugInfo<RescalingPassOutput, LevelingOptimizerOutput,
                                  ScaledTOp, LeveledTOp>(source_pass,
                                                         destination_pass));
  } else if (Estd::contains(parsers, source_pass) &&
             Estd::contains(leveling_optimizers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<ParserOutput, LevelingOptimizerOutput, TOpEmbrio,
                           LeveledTOp>(
        stream,
        compiled_program.GetDebugInfo<ParserOutput, LevelingOptimizerOutput,
                                      TOpEmbrio, LeveledTOp>(source_pass,
                                                             destination_pass));
  } else if (Estd::contains(parsers, source_pass) &&
             Estd::contains(ct_op_optimizers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<ParserOutput, CtOpPassOutput, TOpEmbrio, CtOp>(
        stream,
        compiled_program
            .GetDebugInfo<ParserOutput, CtOpPassOutput, TOpEmbrio, CtOp>(
                source_pass, destination_pass));
  } else if (Estd::contains(leveling_optimizers, source_pass) &&
             Estd::contains(ct_op_optimizers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<LevelingOptimizerOutput, CtOpPassOutput, LeveledTOp,
                           CtOp>(
        stream,
        compiled_program
            .GetDebugInfo<LevelingPassOutput, CtOpPassOutput, LeveledTOp, CtOp>(
                source_pass, destination_pass));
  } else if (Estd::contains(layout_optimizers, source_pass) &&
             Estd::contains(ct_op_optimizers, destination_pass)) {
    auto compiled_program =
        ReadFile<CompiledProgram>(exe_folder / kCompiledProgram);
    DagToDotfileWithGroups<LayoutOptimizerOutput, CtOpPassOutput, TOp, CtOp>(
        stream, compiled_program
                    .GetDebugInfo<LayoutPassOutput, CtOpPassOutput, TOp, CtOp>(
                        source_pass, destination_pass));
  } else {
    LOG(FATAL) << "Unrecognized source or destination pass!";
  }
}
