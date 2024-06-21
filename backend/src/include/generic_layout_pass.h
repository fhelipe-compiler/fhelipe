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

#ifndef FHELIPE_GENERIC_LAYOUT_PASS_H_
#define FHELIPE_GENERIC_LAYOUT_PASS_H_

#include <tuple>

#include "pass_utils.h"
#include "program_context.h"
#include "t_op_embrio.h"
#include "tensor_layout.h"

namespace fhelipe {

class GenericLayoutPass : public LayoutPass {
 public:
  explicit GenericLayoutPass(const ProgramContext& context,
                             bool ignore_chet_repack)
      : context_(context), ignore_chet_repack_(ignore_chet_repack) {}

  LayoutPassOutput DoPass(const LayoutPassInput& in_dag) final;
  virtual TensorLayout GetOutputLayout(
      const std::shared_ptr<Node<TOpEmbrio>>& embrio,
      const TensorLayout& input_layout) = 0;
  virtual std::vector<std::shared_ptr<Node<TOp>>> MatchLayouts(
      Dag<TOp>& dag,
      const std::vector<std::shared_ptr<Node<TOp>>>& parents) = 0;
  virtual TensorLayout DefaultLayout(const Shape& shape,
                                     ChunkSize chunk_size) const = 0;

  const ProgramContext& Context() const { return context_; }

 private:
  ProgramContext context_;
  bool ignore_chet_repack_;

  std::shared_ptr<Node<TOp>> BuildNewNode(
      Dag<TOp>& dag, const std::shared_ptr<Node<TOpEmbrio>>& old_node,
      std::vector<std::shared_ptr<Node<TOp>>> parents, ChunkSize chunk_size);
  TensorLayout NodeLayout(
      const TOpEmbrio& old_node,
      const std::vector<std::shared_ptr<Node<TOp>>>& parents,
      ChunkSize chunk_size) const;
};

}  // namespace fhelipe

#endif  // FHELIPE_GENERIC_LAYOUT_PASS_H_
