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

#include "laid_out_tensor.h"
#include "t_op.h"

namespace fhelipe {

class CtOp;
class RawShiftBit;

namespace ct_program {
class CtProgram;
}  // namespace ct_program

TOp::LaidOutTensorCt DoRawShift(ct_program::CtProgram& ct_dag,
                                const TOp::LaidOutTensorCt& input_tensor,
                                const RawShiftBit& shift_bit);
bool WrapsAround(const RawShiftBit& shift_bit, const TensorIndex& offset);

}  // namespace fhelipe