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

#include "include/translation_mask_utils.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "include/chunk_ir.h"
#include "include/ct_op.h"
#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/laid_out_chunk.h"
#include "include/laid_out_tensor.h"
#include "include/tensor_layout.h"
#include "include/translation_mask_generator.h"

namespace {

using namespace fhelipe;

std::vector<TOp::LaidOutChunk> SumCts(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutChunk>& lhs,
    const std::vector<TOp::LaidOutChunk>& rhs) {
  return Estd::transform(
      lhs, rhs, [&ct_program](const auto& ct0, const auto& ct1) {
        CHECK(ct0.Offset() == ct1.Offset());
        CHECK(ct0.Layout() == ct1.Layout());
        if (dynamic_cast<const ZeroC*>(&ct0.Chunk()->Value())) {
          return ct1;
        }
        if (dynamic_cast<const ZeroC*>(&ct1.Chunk()->Value())) {
          return ct0;
        }
        return TOp::LaidOutChunk{
            ct0.Layout(), ct0.Offset(),
            ct_program::CreateAddCC(ct_program, ct0.Chunk(), ct1.Chunk())};
      });
}

}  // namespace

namespace fhelipe {

LaidOutTensor<ChunkIr> MaskAllInvalidSlots(const TensorLayout& layout) {
  std::vector<TranslationMask> translation_mask = MakeTranslationMasks(
      layout, layout, [](const TensorIndex& ti) { return ti; });
  for (const auto& [rotate_by, mask] : translation_mask) {
    if (rotate_by.ChunkIndexDiff() == 0 && rotate_by.ChunkNumberDiff() == 0) {
      return mask;
    }
  }
  LOG(FATAL);
}

TOp::LaidOutTensorCt ZeroOutWhereZeroMask(ct_program::CtProgram& ct_program,
                                          const TOp::LaidOutTensorCt& ct,
                                          const LaidOutTensor<ChunkIr>& pt) {
  auto zero_c =
      ct_program::FetchZeroCAtSameLevelInfoAs(ct.Chunks().at(0).Chunk());
  auto chunks = Estd::transform(
      ct.Chunks(), pt.Chunks(),
      [&zero_c](const TOp::LaidOutChunk& lhs,
                const LaidOutChunk<ChunkIr>& rhs) {
        CHECK(lhs.Offset() == rhs.Offset());
        CHECK(lhs.Layout() == rhs.Layout());
        // TODO(nsamar): We almost surely do not want the scale
        // of backend-generated masks to be
        // ct_program.GetProgramContext().LogScale(). Because a
        // mask is all 1s and 0s, this scale can be much smaller
        // and still yield good precision.
        auto chunk = std::holds_alternative<ZeroChunkIr>(rhs.Chunk())
                         ? zero_c
                         : lhs.Chunk();

        return TOp::LaidOutChunk{lhs.Layout(), lhs.Offset(), chunk};
      });
  return TOp::LaidOutTensorCt{chunks};
}

TOp::LaidOutTensorCt ApplyMask(ct_program::CtProgram& ct_program,
                               const TOp::LaidOutTensorCt& ct,
                               const LaidOutTensor<ChunkIr>& pt) {
  auto zero_c = ct_program::FetchZeroCThatIsAtSameLevelInfoAsAMulCPChildOf(
      ct.Chunks().at(0).Chunk(), ct_program.GetProgramContext().LogScale());
  auto chunks = Estd::transform(
      ct.Chunks(), pt.Chunks(),
      [&ct_program, &zero_c](const TOp::LaidOutChunk& lhs,
                             const LaidOutChunk<ChunkIr>& rhs) {
        CHECK(lhs.Offset() == rhs.Offset());
        CHECK(lhs.Layout() == rhs.Layout());
        // TODO(nsamar): We almost surely do not want the scale
        // of backend-generated masks to be
        // ct_program.GetProgramContext().LogScale(). Because a
        // mask is all 1s and 0s, this scale can be much smaller
        // and still yield good precision.
        auto chunk = std::holds_alternative<ZeroChunkIr>(rhs.Chunk())
                         ? zero_c
                         : ct_program::CreateMulCP(
                               ct_program, lhs.Chunk(), rhs.Chunk(),
                               ct_program.GetProgramContext().LogScale());

        return TOp::LaidOutChunk{lhs.Layout(), lhs.Offset(), chunk};
      });
  return TOp::LaidOutTensorCt{chunks};
}

std::vector<TOp::LaidOutChunk> ApplyRotation(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutChunk>& cts, int rotate_by) {
  return Estd::transform(cts, [&](const auto& chunk) {
    return dynamic_cast<const ZeroC*>(&chunk.Chunk()->Value())
               ? chunk
               : TOp::LaidOutChunk(chunk.Layout(), chunk.Offset(),
                                   ct_program::CreateRotateC(
                                       ct_program, chunk.Chunk(), rotate_by));
  });
}

std::vector<TOp::LaidOutChunk> PermuteChunks(
    const std::vector<TOp::LaidOutChunk>& ct_ops, int chunk_delta,
    const TensorLayout& output_layout) {
  int output_chunk_count = output_layout.TotalChunks();
  std::vector<TOp::Chunk> result =
      Estd::transform(ct_ops, [](const auto& chunk) { return chunk.Chunk(); });
  while (result.size() < output_chunk_count) {
    result.push_back(ct_program::FetchZeroCAtSameLevelInfoAs(result.at(0)));
  }
  std::rotate(result.begin(), result.end() - chunk_delta, result.end());
  std::vector<TOp::Chunk> truncated_result(result.begin(),
                                           result.begin() + output_chunk_count);

  std::vector<TOp::LaidOutChunk> laid_out_result;
  for (int idx : Estd::indices(truncated_result.size())) {
    laid_out_result.emplace_back(output_layout,
                                 output_layout.ChunkOffsets()[idx],
                                 truncated_result[idx]);
  }
  return laid_out_result;
}

std::vector<TOp::LaidOutChunk> ZeroLaidOutTensor(const TOp::Chunk& sister_node,
                                                 const TensorLayout& layout) {
  auto zero = ct_program::FetchZeroCAtSameLevelInfoAs(sister_node);
  return Estd::transform(layout.ChunkOffsets(),
                         [&layout, &zero](const auto& offset) {
                           return TOp::LaidOutChunk{layout, offset, zero};
                         });
}

std::vector<TOp::LaidOutChunk> ApplyTranslationMasks(
    ct_program::CtProgram& ct_program, const TOp::LaidOutTensorCt& input_tensor,
    const std::vector<TranslationMask>& trans_masks,
    const TensorLayout& output_layout) {
  std::vector<TOp::LaidOutChunk> sum =
      ZeroLaidOutTensor(input_tensor.Chunks().at(0).Chunk(), output_layout);
  auto parent = input_tensor.Chunks().at(0).Chunk();
  for (const auto& [translation, mask_tensor] : trans_masks) {
    auto result = ApplyMask(ct_program, input_tensor, mask_tensor).Chunks();
    auto opt_result =
        PermuteChunks(result, translation.ChunkNumberDiff(), output_layout);
    opt_result =
        ApplyRotation(ct_program, opt_result, translation.ChunkIndexDiff());
    sum = SumCts(ct_program, sum, opt_result);
  }
  return sum;
}

std::vector<TOp::LaidOutChunk> ApplyTranslationsButNotMasks(
    ct_program::CtProgram& ct_program, const TOp::LaidOutTensorCt& input_tensor,
    const std::vector<TranslationMask>& trans_masks,
    const TensorLayout& output_layout) {
  std::vector<TOp::LaidOutChunk> sum =
      ZeroLaidOutTensor(input_tensor.Chunks().at(0).Chunk(), output_layout);
  auto parent = input_tensor.Chunks().at(0).Chunk();
  for (const auto& [translation, mask_tensor] : trans_masks) {
    auto result =
        ZeroOutWhereZeroMask(ct_program, input_tensor, mask_tensor).Chunks();
    auto opt_result =
        PermuteChunks(result, translation.ChunkNumberDiff(), output_layout);
    opt_result =
        ApplyRotation(ct_program, opt_result, translation.ChunkIndexDiff());
    sum = SumCts(ct_program, sum, opt_result);
  }
  return sum;
}

std::vector<TranslationMask> MakeTranslationMasks(
    const TensorLayout& input_layout, const TensorLayout& output_layout,
    const std::function<std::optional<TensorIndex>(const TensorIndex&)>&
        src_to_dest_func) {
  auto trans_mask_gen = TranslationMaskGenerator(input_layout);
  Shape shape = input_layout.GetShape();
  for (int flat_idx = 0; flat_idx < shape.ValueCnt(); ++flat_idx) {
    TensorIndex src_ti = TensorIndex(input_layout.GetShape(), flat_idx);
    std::optional<TensorIndex> dest_ti = src_to_dest_func(src_ti);
    if (!dest_ti.has_value()) {
      continue;
    }
    auto src = LaidOutTensorIndex{input_layout, src_ti};
    auto dest = LaidOutTensorIndex{output_layout, dest_ti.value()};
    trans_mask_gen.RegisterTranslation(TranslationSrcDest(src, dest), src);
  }
  return trans_mask_gen.GetTranslationMasks();
}

}  // namespace fhelipe
