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

#ifndef FHELIPE_FILL_GAPS_LAYOUT_PASS_H_
#define FHELIPE_FILL_GAPS_LAYOUT_PASS_H_

#include <unordered_map>
#include <vector>

#include "debug_info.h"
#include "generic_layout_pass.h"
#include "pass_utils.h"
#include "program_context.h"

namespace fhelipe {

class FillGapsLayoutPass : public GenericLayoutPass {
 public:
  explicit FillGapsLayoutPass(const ProgramContext& context)
      : GenericLayoutPass(context, true) {}
  const PassName& GetPassName() const final {
    static PassName pass_name = PassName("fill_gaps_layout_pass");
    return pass_name;
  }
  std::unique_ptr<LayoutPass> CloneUniq() const final {
    return std::make_unique<FillGapsLayoutPass>(*this);
  }

  static std::vector<TensorLayout::LayoutBit> DefaultLayoutBits(
      const Shape& shape);
  static TensorLayout GetTStrideCOutputLayout(
      const TensorLayout& input_layout, const std::vector<Stride>& strides);
  static TensorLayout GetTResizeDimsCOutputLayout(
      const TensorLayout& input_layout, const Shape& output_shape);
  static TensorLayout GetTReduceDimCOutputLayout(
      const TensorLayout& input_layout, int dimension);
  static TensorLayout GetTReplicateDimCOutputLayout(
      const TensorLayout& input_layout, int dimension, int multiple);
  static TensorLayout GetTReorderDimsCOutputLayout(
      const TensorLayout& input_layout, const std::vector<int>& dim_order);

  static TensorLayout StaticGetOutputLayout(
      const std::shared_ptr<Node<TOpEmbrio>>& embrio,
      const TensorLayout& input_layout);
  static TensorLayout InverseStaticGetOutputLayout(
      const TOp& t_op, const TensorLayout& output_layout);
  TensorLayout GetOutputLayout(const std::shared_ptr<Node<TOpEmbrio>>& embrio,
                               const TensorLayout& input_layout) final {
    return StaticGetOutputLayout(embrio, input_layout);
  }
  std::vector<std::shared_ptr<Node<TOp>>> MatchLayouts(
      Dag<TOp>& dag,
      const std::vector<std::shared_ptr<Node<TOp>>>& parents) final;
  TensorLayout DefaultLayout(const Shape& shape,
                             ChunkSize chunk_size) const final;

 private:
};

}  // namespace fhelipe

#endif  // FHELIPE_FILL_GAPS_LAYOUT_PASS_H_
