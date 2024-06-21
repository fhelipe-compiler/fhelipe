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

#include "include/t_rotate_c.h"

#include <glog/logging.h>

#include <string>
#include <vector>

#include "include/chunk.h"
#include "include/ct_op.h"
#include "include/ct_program.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_index.h"
#include "include/node.h"
#include "include/shape.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_generator.h"
#include "include/translation_mask_utils.h"
#include "include/zero_c.h"

namespace fhelipe {

namespace {

void SanityCheckTRotateC(const TensorLayout& input_layout,
                         const TensorLayout& output_layout) {
  CHECK(input_layout == output_layout);
  CHECK(input_layout.TotalChunks() == 1);
}

}  // namespace

TOp::LaidOutTensorCt TRotateC::AmendCtProgram(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutTensorCt>& input_tensors) const {
  CHECK(input_tensors.size() == 1);
  CHECK(input_tensors[0].Layout() == layout_);
  const auto& chunk = input_tensors.at(0).Chunks().at(0);
  auto result =
      dynamic_cast<const ZeroC*>(&chunk.Chunk()->Value())
          ? chunk
          : TOp::LaidOutChunk(chunk.Layout(), chunk.Offset(),
                              ct_program::CreateRotateC(
                                  ct_program, chunk.Chunk(), rotate_by_));
  return TOp::LaidOutTensorCt{{result}};
}

void TRotateC::SetLayouts(const TensorLayout& input_layout,
                          const TensorLayout& output_layout) {
  SanityCheckTRotateC(input_layout, output_layout);
  layout_ = input_layout;
}

bool TRotateC::EqualTo(const TOp& other) const {
  const auto* t_rotate_c = dynamic_cast<const TRotateC*>(&other);
  return t_rotate_c && t_rotate_c->OutputLayout() == OutputLayout() &&
         t_rotate_c->RotateBy() == RotateBy();
}

TRotateC::TRotateC(const TensorLayout& layout, int rotate_by)
    : layout_(layout), rotate_by_(rotate_by) {
  SanityCheckTRotateC(layout, layout);
}

}  // namespace fhelipe
