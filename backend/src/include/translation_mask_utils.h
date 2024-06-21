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

#ifndef FHELIPE_TRANSLATION_MASK_UTILS_H_
#define FHELIPE_TRANSLATION_MASK_UTILS_H_

#include <vector>

#include "laid_out_tensor.h"
#include "laid_out_tensor_index.h"
#include "t_op.h"
#include "translation_mask_generator.h"

namespace fhelipe {

class CtOp;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

std::vector<TOp::LaidOutChunk> ApplyTranslationMasks(
    ct_program::CtProgram& ct_program, const TOp::LaidOutTensorCt& input_tensor,
    const std::vector<TranslationMask>& trans_masks,
    const TensorLayout& output_layout);

std::vector<TOp::LaidOutChunk> ApplyRotation(
    ct_program::CtProgram& ct_program,
    const std::vector<TOp::LaidOutChunk>& cts, int rotate_by);

std::vector<TOp::LaidOutChunk> ZeroLaidOutTensor(const TOp::Chunk& sister_node,
                                                 const TensorLayout& layout);

std::vector<TOp::LaidOutChunk> ApplyTranslationsButNotMasks(
    ct_program::CtProgram& ct_program, const TOp::LaidOutTensorCt& input_tensor,
    const std::vector<TranslationMask>& trans_masks,
    const TensorLayout& output_layout);

std::vector<TranslationMask> MakeTranslationMasks(
    const TensorLayout& input_layout, const TensorLayout& output_layout,
    const std::function<std::optional<TensorIndex>(const TensorIndex&)>&
        src_to_dest_func);

}  // namespace fhelipe

#endif  // FHELIPE_TRANSLATION_MASK_UTILS_H_
