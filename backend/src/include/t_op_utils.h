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

#ifndef FHELIPE_T_OP_UTILS_H_
#define FHELIPE_T_OP_UTILS_H_

#include <functional>
#include <vector>

#include "chunk_ir.h"
#include "ct_op.h"
#include "ct_program.h"
#include "dictionary.h"
#include "laid_out_tensor.h"
#include "t_op.h"

namespace fhelipe {

TOp::LaidOutTensorCt CreateCtPtTensorOp(
    ct_program::CtProgram& ct_program, const TOp::LaidOutTensorCt& input_tensor,
    const KeyType& frontend_tensor_name, LogScale pt_tensor_log_scale,
    const std::function<TOp::Chunk(ct_program::CtProgram&, const TOp::Chunk&,
                                   const ChunkIr&, LogScale)>& CreateCtOp);

}  // namespace fhelipe

#endif  // FHELIPE_T_OP_UTILS_H_
