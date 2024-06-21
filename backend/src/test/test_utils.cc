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

#include "test/test_utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <random>

#include "gtest/gtest.h"

#include "include/array.h"
#include "include/chunk_size.h"
#include "include/constants.h"
#include "include/extended_std.h"
#include "include/fill_gaps_layout_pass.h"
#include "include/layout_utils.h"
#include "include/maybe_tensor_index.h"
#include "include/packer.h"
#include "include/plaintext.h"
#include "include/ram_dictionary.h"
#include "include/t_add_cp.h"
#include "include/t_input_c.h"
#include "include/t_mul_cp.h"
#include "include/t_output_c.h"
#include "include/utils.h"
#include "test/test_constants.h"

using namespace fhelipe;

RamDictionary<Shape> GatherFrontendTensors(const Dag<TOp>& top_dag) {
  RamDictionary<Shape> result;
  for (const auto& node : top_dag.NodesInTopologicalOrder()) {
    if (const auto* ptr = dynamic_cast<const TAddCP*>(&node->Value())) {
      result.Record(ptr->PtTensorName(), ptr->OutputLayout().GetShape());
    } else if (const auto* ptr = dynamic_cast<const TMulCP*>(&node->Value())) {
      result.Record(ptr->PtTensorName(), ptr->OutputLayout().GetShape());
    }
  }
  return result;
}

fhelipe::Tensor<std::optional<fhelipe::PtVal>> ToOptionalTensor(
    const fhelipe::Tensor<fhelipe::PtVal>& tensor) {
  return {tensor.GetShape(),
          Estd::transform(tensor.Values(),
                          [](const auto& x) { return std::make_optional(x); })};
}

void MakeOutputNode(Dag<TOp>& top_dag, const std::shared_ptr<Node<TOp>>& parent,
                    const std::string& tensor_name) {
  CHECK(parent);
  top_dag.AddNode(std::make_unique<fhelipe::TOutputC>(
                      parent->Value().OutputLayout(), tensor_name),
                  {parent});
}

void DoChecks(const LaidOutTensorDictionary<PtChunk>& outputs,
              const Dictionary<Tensor<std::optional<PtVal>>>& check) {
  for (const auto& tensor_name : check.Keys()) {
    VerifyInvalidSlotsAreZero(outputs.At(tensor_name));
    TestEqualVectorsWithDontCares(Unpack(outputs.At(tensor_name)).Values(),
                                  check.At(tensor_name).Values());
  }
}

std::shared_ptr<fhelipe::Node<fhelipe::TOp>> MakeInputNode(
    Dag<TOp>& top_dag, const TensorLayout& layout,
    const std::string& input_name) {
  return top_dag.AddInput(std::make_unique<TInputC>(
      layout, input_name, kDefaultTestContext.LogScale()));
}

RamDictionary<EncryptionConfig> GetEncryptionConfigs(const Dag<TOp>& top_dag) {
  auto configs = RamDictionary<EncryptionConfig>();
  AddEncryptionConfigs(configs, top_dag);
  return configs;
}

Shape RandomShape(int dimension_count) {
  CHECK(dimension_count > 0);
  int remaining = kMaxTensorElements;
  Array dimension_sizes;
  int target = std::pow(remaining, 1.0 / dimension_count);
  // Generates approximately equal-sized dimensions
  while (dimension_sizes.size() < dimension_count) {
    int new_dimension = 1 + (rand() % std::min(2 * target, remaining));
    dimension_sizes.push_back(new_dimension);
    remaining /= new_dimension;
  }
  return {dimension_sizes};
}

Shape RandomShape() {
  int dimension_count = 1 + (rand() % kMaxDimensionCount);
  return RandomShape(dimension_count);
}

ChunkSize RandomChunkSize() { return 1 << (rand() % kLogMaxSlotSize); }

template <class T>
std::vector<T> Shuffled(std::vector<T> vec) {
  auto rng = std::default_random_engine{};
  std::shuffle(vec.begin(), vec.end(), rng);
  return vec;
}

void VerifyInvalidSlotsAreZero(const LaidOutTensor<PtChunk>& t) {
  const auto& layout = t.Layout();
  const auto& t_chunks = t.Chunks();
  const auto& chunk_offsets = layout.ChunkOffsets();
  for (int chunk_num = 0; chunk_num < layout.TotalChunks(); ++chunk_num) {
    const auto chunk_offset = chunk_offsets[chunk_num];
    const auto& tensor_indices = layout.TensorIndices(chunk_offset);
    for (int chunk_idx = 0; chunk_idx < layout.ChunkSize().value();
         ++chunk_idx) {
      const auto mti = tensor_indices[chunk_idx];
      if (!mti.has_value()) {
        ASSERT_EQ(t_chunks[chunk_num].Chunk().Values()[chunk_idx], 0);
      }
    }
  }
}

std::vector<TensorLayout::LayoutBit> ShuffledWithGaps(
    std::vector<TensorLayout::LayoutBit> vec, int max_gaps) {
  auto rng = std::default_random_engine{};
  int num_gaps = std::rand() % max_gaps;
  for (int i = 0; i < num_gaps; ++i) {
    vec.push_back(std::nullopt);
  }
  std::shuffle(vec.begin(), vec.end(), rng);
  return vec;
}

std::vector<int> ShuffledIota(int max_val) {
  std::vector<int> result(max_val);
  std::iota(result.begin(), result.end(), 0);
  auto rng = std::default_random_engine{};
  std::shuffle(result.begin(), result.end(), rng);
  return result;
}

bool AllChunkBitsNullopt(
    const std::vector<TensorLayout::LayoutBit>& layout_bits,
    int log2_chunk_size) {
  CHECK(log2_chunk_size <= layout_bits.size());
  return std::all_of(
      layout_bits.begin(), layout_bits.begin() + log2_chunk_size + 1,
      [&](const TensorLayout::LayoutBit& lb) { return !lb.has_value(); });
}

fhelipe::TensorLayout RandomLayoutWithoutGaps(const fhelipe::Shape& shape,
                                             int chunk_size) {
  std::vector<TensorLayout::LayoutBit> layout_bits =
      Shuffled(FillGapsLayoutPass::DefaultLayoutBits(shape));
  return {shape, ChunkBits(layout_bits, chunk_size)};
}

fhelipe::TensorLayout RandomLayout(const fhelipe::Shape& shape,
                                  ChunkSize chunk_size) {
  std::vector<TensorLayout::LayoutBit> layout_bits = ShuffledWithGaps(
      FillGapsLayoutPass::DefaultLayoutBits(shape), kMaxTestLayoutGaps);
  return {shape, ChunkBits(layout_bits, chunk_size)};
}

fhelipe::TensorLayout RandomLayout(const fhelipe::Shape& shape) {
  return RandomLayout(shape, RandomChunkSize());
}

TensorLayout RandomLayout() {
  Shape shape = RandomShape();
  return RandomLayout(shape);
}

fhelipe::PtVal LInfinityNorm(const std::vector<fhelipe::PtVal>& values) {
  return Estd::max_element(Estd::transform(
      values, [](fhelipe::PtVal value) { return std::abs(value); }));
}

LaidOutTensor<PtChunk> ToyTensor(const fhelipe::TensorLayout& layout) {
  std::vector<PtVal> vec(layout.GetShape().ValueCnt());
  std::iota(vec.begin(), vec.end(), 0);
  return Pack(vec, layout);
}

Tensor<PtVal> MakeTensor(const Shape& shape) {
  std::vector<PtVal> values =
      Estd::transform(Estd::indices(shape.ValueCnt()),
                      [](const auto& x) { return static_cast<PtVal>(x); });
  return {shape, values};
}

fhelipe::RamDictionary<Tensor<PtVal>> MakeFrontendTensors(
    const Dictionary<Shape>& frontend_tensors) {
  RamDictionary<Tensor<PtVal>> result;
  for (const auto& tensor_name : frontend_tensors.Keys()) {
    result.Record(tensor_name, MakeTensor(frontend_tensors.At(tensor_name)));
  }
  return result;
}

DiffTensorIndex RandomDiffTensorIndex(const Shape& shape) {
  return {shape, RandomTensorIndex(shape).DimensionIndices()};
}

TensorIndex RandomTensorIndex(const Shape& shape) {
  Array indices;
  for (int max_value : shape) {
    indices.push_back(rand() % max_value);
  }
  return {shape, indices};
}
