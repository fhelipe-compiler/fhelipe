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

#include "include/compiler.h"

#include "include/compiled_program.h"
#include "include/encryption_config.h"
#include "include/pass_utils.h"

namespace fhelipe {

Compiler::Compiler(OwningVector<Preprocessor>&& preprocessors,
                   std::unique_ptr<Parser>&& parser,
                   OwningVector<EmbrioOptimizer>&& embrio_optimizers,
                   std::unique_ptr<LayoutPass>&& layout_pass,
                   OwningVector<LayoutOptimizer>&& layout_optimizers,
                   std::unique_ptr<RescalingPass>&& rescaling_pass,
                   std::unique_ptr<LevelingPass>&& leveling_pass,
                   OwningVector<LevelingOptimizer>&& leveling_optimizers,
                   std::unique_ptr<CtOpPass>&& ct_op_pass,
                   OwningVector<CtOpOptimizer>&& ct_op_optimizers)
    : preprocessors_(std::move(preprocessors)),
      parser_(std::move(parser)),
      embrio_optimizers_(std::move(embrio_optimizers)),
      layout_pass_(std::move(layout_pass)),
      layout_optimizers_(std::move(layout_optimizers)),
      rescaling_pass_(std::move(rescaling_pass)),
      leveling_pass_(std::move(leveling_pass)),
      leveling_optimizers_(std::move(leveling_optimizers)),
      ct_op_pass_(std::move(ct_op_pass)),
      ct_op_optimizers_(std::move(ct_op_optimizers)) {}

CompiledProgram Compiler::Compile(const Program& program) {
  CompiledProgramBuilder builder(this, program);

  auto code = program.Code();

  for (auto& preprocessor : preprocessors_) {
    code = preprocessor->DoPass(code);
  }

  ParserOutput dag = parser_->DoPass(code);
  ParserOutput* dag_ptr =
      builder.RecordPass(parser_->GetPassName(), std::move(dag));

  for (auto& embrio_optimizer : embrio_optimizers_) {
    *dag_ptr = embrio_optimizer->DoPass(*dag_ptr);
  }

  LayoutPassOutput laid_out_dag = layout_pass_->DoPass(*dag_ptr);
  LayoutPassOutput* laid_out_dag_ptr =
      builder.RecordPass(layout_pass_->GetPassName(), std::move(laid_out_dag));
  for (auto& layout_optimizer : layout_optimizers_) {
    laid_out_dag = layout_optimizer->DoPass(*laid_out_dag_ptr);
    laid_out_dag_ptr = builder.RecordPass(layout_optimizer->GetPassName(),
                                          std::move(laid_out_dag));
  }

  // Dump encryption config now, since we know what the layouts are
  auto config_dict = ClearedPersistedDictionary<EncryptionConfig>(
      program.ExeFolder() / kEncCfg);
  AddEncryptionConfigs(config_dict, *laid_out_dag_ptr);

  RescalingPassOutput rescaled_dag = rescaling_pass_->DoPass(*laid_out_dag_ptr);
  RescalingPassOutput* rescaled_dag_ptr = builder.RecordPass(
      rescaling_pass_->GetPassName(), std::move(rescaled_dag));

  LevelingPassOutput leveled_dag = leveling_pass_->DoPass(*rescaled_dag_ptr);
  LevelingPassOutput* leveled_dag_ptr =
      builder.RecordPass(leveling_pass_->GetPassName(), std::move(leveled_dag));
  for (auto& leveling_optimizer : leveling_optimizers_) {
    leveled_dag = leveling_optimizer->DoPass(*leveled_dag_ptr);
    leveled_dag_ptr = builder.RecordPass(leveling_optimizer->GetPassName(),
                                         std::move(leveled_dag));
  }

  CtOpPassOutput ct_op_dag = ct_op_pass_->DoPass(*leveled_dag_ptr);
  CtOpPassOutput* ct_op_dag_ptr =
      builder.RecordPass(ct_op_pass_->GetPassName(), std::move(ct_op_dag));
  for (auto& ct_op_optimizer : ct_op_optimizers_) {
    ct_op_dag = ct_op_optimizer->DoPass(*ct_op_dag_ptr);
    ct_op_dag_ptr = builder.RecordPass(ct_op_optimizer->GetPassName(),
                                       std::move(ct_op_dag));
  }

  return builder.GetCompiledProgram();
}

Compiler CompilerBuilder::GetCompiler() {
  return {std::move(preprocessors_),     std::move(parser_),
          std::move(embrio_optimizers_), std::move(layout_pass_),
          std::move(layout_optimizers_), std::move(rescaling_pass_),
          std::move(leveling_pass_),     std::move(leveling_optimizers_),
          std::move(ct_op_pass_),        std::move(ct_op_optimizers_)};
}

}  // namespace fhelipe
