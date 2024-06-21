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

#include "include/dag_io.h"

#include <math.h>

#include "include/constants.h"
#include "include/ct_op.h"
#include "include/input_c.h"
#include "include/leveled_t_op.h"
#include "include/node.h"
#include "include/scaled_pt_val.h"
#include "include/scaled_t_op.h"
#include "include/t_op.h"
#include "include/t_op_embrio.h"

namespace fhelipe {

template <>
Dag<TOpEmbrio> ReadStream<Dag<TOpEmbrio>>(std::istream& stream) {
  return details::ReadDag<TOpEmbrio>(stream, &TOpEmbrio::CreateInstance);
}

template <>
Dag<TOp> ReadStream<Dag<TOp>>(std::istream& stream) {
  return details::ReadDag<TOp>(stream, &TOp::CreateInstance);
}

template <>
Dag<CtOp> ReadStream<Dag<CtOp>>(std::istream& stream) {
  return details::ReadDag<CtOp>(stream, &CtOp::CreateInstance);
}
template <>
void WriteStream<Dag<CtOp>>(std::ostream& stream, const Dag<CtOp>& dag) {
  details::WriteDag<CtOp>(stream, dag);
}

}  // namespace fhelipe
