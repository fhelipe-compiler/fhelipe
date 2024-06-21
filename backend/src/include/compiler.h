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

#ifndef FHELIPE_COMPILER_H_
#define FHELIPE_COMPILER_H_

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "compiled_program.h"
#include "leveled_t_op.h"
#include "pass.h"
#include "pass_utils.h"
#include "scaled_t_op.h"
#include "t_op.h"
#include "t_op_embrio.h"

namespace fhelipe {

class Compiler {
 public:
  Compiler(OwningVector<Preprocessor>&& preprocessors,
           std::unique_ptr<Parser>&& parser,
           OwningVector<EmbrioOptimizer>&& embrio_optimizers,
           std::unique_ptr<LayoutPass>&& layout_pass,
           OwningVector<LayoutOptimizer>&& layout_optimizers,
           std::unique_ptr<RescalingPass>&& rescaling_pass,
           std::unique_ptr<LevelingPass>&& leveling_pass,
           OwningVector<LevelingOptimizer>&& leveling_optimizers,
           std::unique_ptr<CtOpPass>&& ct_op_pass,
           OwningVector<CtOpOptimizer>&& ct_op_optimizers);

  CompiledProgram Compile(const Program& program);

 private:
  OwningVector<Preprocessor> preprocessors_;
  std::unique_ptr<Parser> parser_;
  OwningVector<EmbrioOptimizer> embrio_optimizers_;
  std::unique_ptr<LayoutPass> layout_pass_;
  OwningVector<LayoutOptimizer> layout_optimizers_;
  std::unique_ptr<RescalingPass> rescaling_pass_;
  std::unique_ptr<LevelingPass> leveling_pass_;
  OwningVector<LevelingOptimizer> leveling_optimizers_;
  std::unique_ptr<CtOpPass> ct_op_pass_;
  OwningVector<CtOpOptimizer> ct_op_optimizers_;
};

class CompilerBuilder {
 public:
  template <typename PassT>
  void AddPass(const PassT& pass) = delete;

  Compiler GetCompiler();

 private:
  OwningVector<Preprocessor> preprocessors_;
  std::unique_ptr<Parser> parser_;
  OwningVector<EmbrioOptimizer> embrio_optimizers_;
  std::unique_ptr<LayoutPass> layout_pass_;
  OwningVector<LayoutOptimizer> layout_optimizers_;
  std::unique_ptr<RescalingPass> rescaling_pass_;
  std::unique_ptr<LevelingPass> leveling_pass_;
  OwningVector<LevelingOptimizer> leveling_optimizers_;
  std::unique_ptr<CtOpPass> ct_op_pass_;
  OwningVector<CtOpOptimizer> ct_op_optimizers_;
};

template <>
inline void CompilerBuilder::AddPass<Preprocessor>(const Preprocessor& pass) {
  preprocessors_.emplace_back(pass.CloneUniq());
}

template <>
inline void CompilerBuilder::AddPass<EmbrioOptimizer>(
    const EmbrioOptimizer& pass) {
  embrio_optimizers_.emplace_back(pass.CloneUniq());
}

template <>
inline void CompilerBuilder::AddPass<Parser>(const Parser& pass) {
  CHECK(!parser_);
  parser_ = pass.CloneUniq();
}

template <>
inline void CompilerBuilder::AddPass<LayoutPass>(const LayoutPass& pass) {
  CHECK(!layout_pass_);
  layout_pass_ = pass.CloneUniq();
}

template <>
inline void CompilerBuilder::AddPass<LayoutOptimizer>(
    const LayoutOptimizer& pass) {
  layout_optimizers_.emplace_back(pass.CloneUniq());
}

template <>
inline void CompilerBuilder::AddPass<RescalingPass>(const RescalingPass& pass) {
  CHECK(!rescaling_pass_);
  rescaling_pass_ = pass.CloneUniq();
}

template <>
inline void CompilerBuilder::AddPass<LevelingPass>(const LevelingPass& pass) {
  CHECK(!leveling_pass_);
  leveling_pass_ = pass.CloneUniq();
}

template <>
inline void CompilerBuilder::AddPass<LevelingOptimizer>(
    const LevelingOptimizer& pass) {
  leveling_optimizers_.emplace_back(pass.CloneUniq());
}

template <>
inline void CompilerBuilder::AddPass<CtOpPass>(const CtOpPass& pass) {
  CHECK(!ct_op_pass_);
  ct_op_pass_ = pass.CloneUniq();
}

template <>
inline void CompilerBuilder::AddPass<CtOpOptimizer>(const CtOpOptimizer& pass) {
  ct_op_optimizers_.emplace_back(pass.CloneUniq());
}

}  // namespace fhelipe

#endif  // FHELIPE_COMPILER_H_
