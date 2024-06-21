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

#include <ostream>
#include <string>

#include "include/basic_ct_op_pass.h"
#include "include/fhebooster_pass.h"
#include "include/basic_parser.h"
#include "include/basic_preprocessor.h"
#include "include/bootstrap_prunning_pass.h"
#include "include/bootstrapping_precision.h"
#include "include/chet_layout_pass.h"
#include "include/compiler.h"
#include "include/constants.h"
#include "include/conversion_decomposer_pass.h"
#include "include/ct_program.h"
#include "include/dag.h"
#include "include/dag_io.h"  // IWYU pragma: keep
#include "include/dp_bootstrapping_pass.h"
#include "include/dummy_ct_op_pass.h"
#include "include/encryption_config.h"
#include "include/filesystem_utils.h"
#include "include/fill_gaps_layout_pass.h"
#include "include/input_layout_pass.h"
#include "include/layout_hoisting_pass.h"
#include "include/lazy_bootstrapping_on_chet_repacks_pass.h"
#include "include/lazy_bootstrapping_pass.h"
#include "include/level_minimization_pass.h"
#include "include/leveled_t_op.h"
#include "include/merge_stride_chain_pass.h"
#include "include/noop_leveling_pass.h"
#include "include/pass_utils.h"
#include "include/persisted_dictionary.h"
#include "include/repack_showering_pass.h"
#include "include/scaled_t_op.h"
#include "include/t_op.h"         // IWYU pragma: keep
#include "include/t_op_embrio.h"  // IWYU pragma: keep
#include "include/value_numbering_pass.h"
#include "include/waterline_rescale.h"
#include "latticpp/ckks/lattigo_param.h"
#include "targets/gflag_utils/exe_folder_gflag_utils.h"
#include "targets/gflag_utils/program_context_gflag_utils.h"
#include "targets/gflag_utils/verbose_gflag_utils.h"

using namespace fhelipe;

namespace {

DEFINE_bool(sched_dfg, false,
            "Set flag to see generate dataflow graph for Axel's scheduler");
DEFINE_string(leveling_pass, "dp",
              "Bootstrapping pass type for the compiler (dp, lazy, or noop)");
DEFINE_string(layout_pass, "fill_gaps",
              "Layout pass type for the compiler (fill_gaps, chet)");
DEFINE_int32(
    max_tentacles_per_conversion, 16,
    "Maximum number of tentacles per layout conversion (must be power of 2)");
DEFINE_string(ct_op_pass, "basic",
              "CtOp pass type for the compiler (basic, dummy)");
DEFINE_bool(repack_shower, false, "Set to true to add repacks to all edges");

std::unique_ptr<LevelingPass> BootstrappingPassFromFlags(
    const ProgramContext& context) {
  if (FLAGS_leveling_pass == "dp") {
    return std::make_unique<DpBootstrappingPass>(context);
  } else if (FLAGS_leveling_pass == "lazy") {
    return std::make_unique<LazyBootstrappingPass>(context);
  } else if (FLAGS_leveling_pass == "noop") {
    return std::make_unique<NoopLevelingPass>(context);
  } else if (FLAGS_leveling_pass == "chet_lazy") {
    return std::make_unique<LazyBootstrappingOnChetRepacksPass>(context);
  }
  LOG(FATAL);
}

std::unique_ptr<LayoutPass> LayoutPassFromFlags(const ProgramContext& context) {
  if (FLAGS_layout_pass == "fill_gaps") {
    return std::make_unique<FillGapsLayoutPass>(context);
  }
  if (FLAGS_layout_pass == "chet") {
    return std::make_unique<ChetLayoutPass>(context);
  }
  LOG(FATAL);
}

std::unique_ptr<CtOpPass> CtOpPassFromFlags(
    const ProgramContext& context,
    std::unique_ptr<PersistedDictionary<ChunkIr>>&& chunk_dict) {
  if (FLAGS_ct_op_pass == "dummy") {
    return std::make_unique<DummyCtOpPass>(context, std::move(chunk_dict));
  }
  if (FLAGS_ct_op_pass == "basic") {
    return std::make_unique<BasicCtOpPass>(context, std::move(chunk_dict));
  }
  LOG(FATAL);
}

bool PruneBootstraps() {
  return FLAGS_leveling_pass == "dp" || FLAGS_leveling_pass == "chet_lazy";
}

}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::ios_base::sync_with_stdio(false);

  auto exe_folder = ExeFolderFromFlags();
  auto context = ProgramContextFromFlags();

  auto builder = CompilerBuilder();

  builder.AddPass<Preprocessor>(
      BasicPreprocessor(context.LogScale(), context.UsableLevels()));
  builder.AddPass<Parser>(BasicParser());
  if (FLAGS_repack_shower) {
    builder.AddPass<EmbrioOptimizer>(RepackShoweringPass());
    ChetLayoutPass::SetRowMajorHack();  // force ChetLayoutPass to use row-major
  }
  builder.AddPass<EmbrioOptimizer>(MergeStrideChainPass());
  builder.AddPass<LayoutPass>(*LayoutPassFromFlags(context));
  if (FLAGS_layout_pass == "fill_gaps") {
    builder.AddPass<LayoutOptimizer>(LayoutHoistingPass());
    builder.AddPass<LayoutOptimizer>(ValueNumberingPass());
    builder.AddPass<LayoutOptimizer>(InputLayoutPass());
    builder.AddPass<LayoutOptimizer>(
        ConversionDecomposerPass(FLAGS_max_tentacles_per_conversion));
  } else if (ChetLayoutPass::RowMajorHack()) {
    builder.AddPass<LayoutOptimizer>(
        ConversionDecomposerPass(FLAGS_max_tentacles_per_conversion));
  }
  builder.AddPass<RescalingPass>(WaterlineRescale(context));
  builder.AddPass<LevelingPass>(*BootstrappingPassFromFlags(context));
  if (PruneBootstraps()) {
    builder.AddPass<LevelingOptimizer>(BootstrapPrunningPass(context));
  }
  builder.AddPass<CtOpPass>(*CtOpPassFromFlags(
      context, std::make_unique<PersistedDictionary<ChunkIr>>(
                   ClearedPersistedDictionary<ChunkIr>(exe_folder / kChIr))));
  if (FLAGS_ct_op_pass != "dummy") {
    builder.AddPass<CtOpOptimizer>(LevelMinimizationPass());
  }

  builder.AddPass<CtOpOptimizer>(FheBoosterPass(context.UsableLevels()));

  auto compiler = builder.GetCompiler();

  auto compiled_program = compiler.Compile(Program(exe_folder));

  LOG(INFO) << "Writting out " << kExecutable;
  WriteFile(exe_folder / kExecutable, compiled_program.Result());

  if (FLAGS_sched_dfg) {
    LOG(INFO) << "Writting out " << kSchedulableDfg;
    auto level_to_craterlake_level_map =
        ct_program::BestPossibleLevelToCraterLakeLevelMap(
            context.UsableLevels(), context.LogScale());
    auto level_to_log_q_map = ct_program::DefaultLevelToLogQMap(
        context.UsableLevels(), context.LogScale());
    auto stream = std::ofstream(exe_folder / kSchedulableDfg);
    WriteSchedulableDataflowGraph(stream, compiled_program.Result(),
                                  level_to_craterlake_level_map,
                                  level_to_log_q_map);
    auto partitioned_programs = PartitionProgram(compiled_program.Result());
    for (int i : Estd::indices(partitioned_programs.size())) {
      auto stream =
          std::ofstream(exe_folder / ("partition" + std::to_string(i)));
      WriteSchedulableDataflowGraph(stream, partitioned_programs.at(i),
                                    level_to_craterlake_level_map,
                                    level_to_log_q_map);
      WriteFile(exe_folder / ("rt_partition" + std::to_string(i)),
                partitioned_programs.at(i));
    }
  }

  WriteFile(exe_folder / kCompiledProgram, compiled_program);
}
