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

#ifndef FHELIPE_PASS_UTILS_H_
#define FHELIPE_PASS_UTILS_H_

#include "extended_std.h"
#include "leveled_t_op.h"
#include "pass.h"
#include "scaled_t_op.h"
#include "t_op.h"
#include "t_op_embrio.h"

namespace fhelipe {

template <typename T>
using OwningVector = std::vector<std::unique_ptr<T>>;

using PreprocessorInput = std::string;
using PreprocessorOutput = std::string;
using Preprocessor = Pass<PreprocessorInput, PreprocessorOutput>;

using ParserInput = PreprocessorOutput;
using ParserOutput = Dag<TOpEmbrio>;
using Parser = Pass<ParserInput, ParserOutput>;

using EmbrioOptimizerInput = ParserOutput;
using EmbrioOptimizerOutput = EmbrioOptimizerInput;
using EmbrioOptimizer = Pass<EmbrioOptimizerInput, EmbrioOptimizerOutput>;

using LayoutPassInput = ParserOutput;
using LayoutPassOutput = Dag<TOp>;
using LayoutPass = Pass<LayoutPassInput, LayoutPassOutput>;

using LayoutOptimizerInput = LayoutPassOutput;
using LayoutOptimizerOutput = LayoutPassOutput;
using LayoutOptimizer = Pass<LayoutOptimizerInput, LayoutOptimizerOutput>;

using RescalingPassInput = LayoutOptimizerOutput;
using RescalingPassOutput = Dag<ScaledTOp>;
using RescalingPass = Pass<RescalingPassInput, RescalingPassOutput>;

using LevelingPassInput = RescalingPassOutput;
using LevelingPassOutput = Dag<LeveledTOp>;
using LevelingPass = Pass<LevelingPassInput, LevelingPassOutput>;

using LevelingOptimizerInput = LevelingPassOutput;
using LevelingOptimizerOutput = Dag<LeveledTOp>;
using LevelingOptimizer = Pass<LevelingOptimizerInput, LevelingOptimizerOutput>;

using CtOpPassInput = LevelingOptimizerOutput;
using CtOpPassOutput = ct_program::CtProgram;
using CtOpPass = Pass<CtOpPassInput, CtOpPassOutput>;

using CtOpOptimizerInput = CtOpPassOutput;
using CtOpOptimizerOutput = CtOpOptimizerInput;
using CtOpOptimizer = Pass<CtOpOptimizerInput, CtOpOptimizerOutput>;

template <typename OldNodeT, typename NewNodeT>
std::vector<std::shared_ptr<Node<NewNodeT>>> ExtractParents(
    const std::unordered_map<const Node<OldNodeT>*,
                             std::shared_ptr<Node<NewNodeT>>>& old_to_new_nodes,
    const Node<OldNodeT>& old_node) {
  return Estd::values_from_keys(old_to_new_nodes, old_node.Parents());
}

}  // namespace fhelipe

#endif  // FHELIPE_PASS_UTILS_H_
