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

#include "include/t_op_utils.h"

#include <functional>
#include <vector>

#include "include/chunk_ir.h"
#include "include/ct_op.h"
#include "include/ct_program.h"
#include "include/dictionary.h"
#include "include/extended_std.h"
#include "include/t_op.h"

namespace fhelipe {

TOp::LaidOutTensorCt CreateCtPtTensorOp(
    ct_program::CtProgram& ct_program, const TOp::LaidOutTensorCt& input_tensor,
    const KeyType& frontend_tensor_name, LogScale pt_tensor_log_scale,
    const std::function<TOp::Chunk(ct_program::CtProgram&, const TOp::Chunk&,
                                   const ChunkIr&, LogScale)>& CreateCtOp) {
  const std::vector<TOp::LaidOutChunk>& ct_ops = input_tensor.Chunks();
  const TensorLayout& layout = input_tensor.Layout();
  const auto& offsets = layout.ChunkOffsets();

  auto result = Estd::transform(
      ct_ops, offsets,
      [&frontend_tensor_name, &layout, &ct_program, &CreateCtOp,
       &pt_tensor_log_scale](const auto& chunk, const TensorIndex& offset) {
        auto flat_indices =
            FlatMaybeTensorIndices(layout.TensorIndices(offset));
        auto ct_op =
            CreateCtOp(ct_program, chunk.Chunk(),
                       IndirectChunkIr(frontend_tensor_name, flat_indices),
                       pt_tensor_log_scale);
        return TOp::LaidOutChunk{chunk.Layout(), chunk.Offset(), ct_op};
      });

  return TOp::LaidOutTensorCt{result};
}

}  // namespace fhelipe
