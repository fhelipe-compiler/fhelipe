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

#include "include/conversion_decomposer_pass.h"

#include <algorithm>
#include <optional>

#include "include/dimension_bit.h"
#include "include/extended_std.h"
#include "include/node.h"
#include "include/permutation.h"
#include "include/t_layout_conversion_c.h"
#include "include/tensor_layout.h"

namespace fhelipe {

namespace {

int MismatchingLayoutBitCount(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) {
  const auto& in_bits = input_layout.Bits();
  const auto& out_bits = output_layout.Bits();
  CHECK(in_bits.size() == out_bits.size());
  return Estd::count_if(
      Estd::indices(in_bits.size()),
      [&in_bits, &out_bits](int idx) { return in_bits[idx] != out_bits[idx]; });
}

bool IsExpensive(const TLayoutConversionC& conversion,
                 int max_tentacles_per_conversion) {
  return (1 << MismatchingLayoutBitCount(conversion.InputLayout(),
                                         conversion.OutputLayout())) >
         max_tentacles_per_conversion;
}

bool IsExpensiveConversion(const TOp& t_op, int max_tentacles_per_conversion) {
  const auto* layout_conversion =
      dynamic_cast<const TLayoutConversionC*>(&t_op);
  return layout_conversion &&
         IsExpensive(*layout_conversion, max_tentacles_per_conversion);
}

int TraceCycleUntilOut(const std::vector<TensorLayout::LayoutBit>& in_bits,
                       const std::vector<TensorLayout::LayoutBit>& out_bits,
                       int idx) {
  while (Estd::contains(out_bits, in_bits[idx])) {
    idx = Estd::find_index(out_bits, in_bits[idx]);
  }
  return idx;
}

std::vector<TensorLayout::LayoutBit> NulloptToNegativeDimension(
    std::vector<TensorLayout::LayoutBit> bits) {
  int fake_idx = 0;
  for (auto& bit : bits) {
    if (bit == std::nullopt) {
      bit = std::make_optional<DimensionBit>(-1, fake_idx++);
    }
  }
  return bits;
}

void MatchNegativeDimensions(std::vector<TensorLayout::LayoutBit>& lhs,
                             std::vector<TensorLayout::LayoutBit>& rhs) {
  for (int idx : Estd::indices(lhs.size())) {
    if (lhs.at(idx)->dimension == -1 && rhs.at(idx)->dimension == -1) {
      if (lhs.at(idx)->bit_index != rhs.at(idx)->bit_index &&
          Estd::contains(rhs, lhs.at(idx))) {
        int swap_idx = Estd::find_index(rhs, lhs.at(idx));
        rhs.at(swap_idx) = rhs.at(idx);
      }
      rhs.at(idx) = lhs.at(idx);
    }
  }
}

std::pair<std::vector<TensorLayout::LayoutBit>,
          std::vector<TensorLayout::LayoutBit>>
ConstructPermutableLayoutBits(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) {
  auto in_bits = NulloptToNegativeDimension(input_layout.Bits());
  auto out_bits = NulloptToNegativeDimension(output_layout.Bits());
  MatchNegativeDimensions(in_bits, out_bits);

  for (int idx : Estd::indices(in_bits.size())) {
    if (!Estd::contains(in_bits, out_bits[idx])) {
      in_bits.push_back(out_bits[idx]);
      int out_idx = TraceCycleUntilOut(in_bits, out_bits, idx);
      out_bits.push_back(in_bits[out_idx]);
    }
  }
  return {in_bits, out_bits};
}

Permutation ExtractPermutation(
    const std::vector<TensorLayout::LayoutBit>& in_bits,
    const std::vector<TensorLayout::LayoutBit>& out_bits) {
  return PermutationFromSourceDestinationPair(in_bits, out_bits);
}

TensorLayout CleanUp(const Shape& shape,
                     const std::vector<TensorLayout::LayoutBit>& bits,
                     const ChunkSize& chunk_size) {
  std::vector<TensorLayout::LayoutBit> result;
  int idx = 0;
  for (auto bit : bits) {
    if (idx++ >= LogChunkSize(chunk_size).value()) {
      break;
    }
    if (bit.value().dimension == -1) {
      result.push_back(std::nullopt);
    } else {
      result.push_back(bit);
    }
  }
  return {shape, result};
}

namespace {

Permutation NextCycleSmallerThanK(std::vector<PermutationCycle>& cycles,
                                  int& max_cycle_size) {
  if (max_cycle_size >= cycles.back().CycleSize()) {
    auto result = cycles.back();
    cycles.pop_back();
    max_cycle_size -= result.CycleSize();
    return result.ToPermutation();
  } else {
    auto [first, second] = cycles.back().BreakUp(max_cycle_size);
    max_cycle_size = 0;
    cycles.pop_back();
    cycles.push_back(second);
    return first.ToPermutation();
  }
}

std::vector<Permutation> BreakUpHelper(std::vector<PermutationCycle> cycles,
                                       int max_non_fixed) {
  if (cycles.empty()) {
    return {};
  }

  Permutation current(cycles.at(0).PermutationSize());
  int remaining = max_non_fixed;
  while (!cycles.empty() && remaining > 1) {
    current = Compose(current, NextCycleSmallerThanK(cycles, remaining));
  }

  return Estd::concat({current}, BreakUpHelper(cycles, max_non_fixed));
}

std::vector<TensorLayout> PermutationsToLayouts(
    const std::vector<TensorLayout::LayoutBit>& start_bits,
    const std::vector<Permutation>& permutations, const Shape& shape,
    const ChunkSize& chunk_size) {
  std::vector<std::vector<TensorLayout::LayoutBit>> bits_sequence{start_bits};
  for (const auto& permutation : permutations) {
    bits_sequence.push_back(permutation.Apply(bits_sequence.back()));
  }

  std::vector<TensorLayout> result;
  for (const auto& bits : bits_sequence) {
    result.push_back(CleanUp(shape, bits, chunk_size));
  }

  return result;
}

void BuildNewNode(const std::shared_ptr<Node<TOp>>& t_op,
                  const std::vector<TensorLayout>& layouts) {
  auto new_node = t_op;
  auto children = t_op->Children();
  auto parents = t_op->Parents();

  for (int idx : Estd::indices(layouts.size() - 1)) {
    new_node = CreateChild<TOp>(std::make_unique<TLayoutConversionC>(
                                    layouts.at(idx), layouts.at(idx + 1)),
                                parents, t_op->Ancestors());
    parents = {new_node};
  }

  for (const auto& child : children) {
    AddParentChildEdge(new_node, child);
    if (child->IsDoubleParent(*t_op)) {
      AddParentChildEdge(new_node, child);
    }
  }
  RemoveNodeWithoutReassaigningChildren(t_op);
}

}  // namespace

std::vector<Permutation> BreakUpIntoPermutationsWithAtLeastKFixedPoints(
    const Permutation& permutation, int k) {
  auto permutation_cycles = permutation.Cycles();
  return BreakUpHelper(permutation_cycles, permutation.Size() - k);
}

void DecomposeConversion(const std::shared_ptr<Node<TOp>>& t_op,
                         int max_bit_permutations_per_conversion) {
  const auto* conversion =
      dynamic_cast<const TLayoutConversionC*>(&t_op->Value());
  CHECK(conversion);

  auto input_layout = conversion->InputLayout();
  auto output_layout = conversion->OutputLayout();
  auto [in_bits, out_bits] =
      ConstructPermutableLayoutBits(input_layout, output_layout);
  auto permutation = ExtractPermutation(in_bits, out_bits);

  std::vector<Permutation> permutations =
      BreakUpIntoPermutationsWithAtLeastKFixedPoints(
          permutation, in_bits.size() - max_bit_permutations_per_conversion);

  auto layouts = PermutationsToLayouts(
      in_bits, permutations, input_layout.GetShape(), input_layout.ChunkSize());

  BuildNewNode(t_op, layouts);
}

}  // namespace

LayoutOptimizerOutput ConversionDecomposerPass::DoPass(
    const LayoutOptimizerInput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag);
  for (const auto& node : out_dag.NodesInTopologicalOrder()) {
    if (IsExpensiveConversion(node->Value(), max_tentacles_per_conversion_)) {
      DecomposeConversion(node, ceil_log2(max_tentacles_per_conversion_));
    }
  }
  return out_dag;
}

}  // namespace fhelipe
