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

#ifndef FHELIPE_LAYOUT_UTILS_H_
#define FHELIPE_LAYOUT_UTILS_H_

#include "chunk_size.h"
#include "dag.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {

std::shared_ptr<Node<TOp>> AddLayoutConversion(
    Dag<TOp>& dag, const std::shared_ptr<Node<TOp>>& node,
    const TensorLayout& output_layout);

std::vector<TensorLayout::LayoutBit> ChunkBits(
    std::vector<TensorLayout::LayoutBit> layout_bits, ChunkSize chunk_size);

std::vector<std::shared_ptr<Node<TOp>>> MatchLayoutsForHoisting(
    Dag<TOp>& dag, const std::vector<std::shared_ptr<Node<TOp>>>& nodes);

bool HasLinearChainToInput(std::shared_ptr<Node<TOp>> node);

bool AllLayoutsMatch(const std::vector<std::shared_ptr<Node<TOp>>>& nodes);

}  // namespace fhelipe

#endif  // FHELIPE_LAYOUT_UTILS_H_
