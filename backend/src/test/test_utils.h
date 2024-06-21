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

#ifndef FHELIPE_TEST_UTILS_H_
#define FHELIPE_TEST_UTILS_H_

#include <glog/logging.h>

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include <gtest/gtest.h>  // IWYU pragma: keep

#include "include/basic_ct_op_pass.h"
#include "include/chunk_size.h"
#include "include/cleartext.h"
#include "include/constants.h"
#include "include/dictionary.h"
#include "include/evaluator.h"
#include "include/io_manager.h"
#include "include/laid_out_tensor.h"
#include "include/laid_out_tensor_dictionary.h"
#include "include/lazy_bootstrapping_pass.h"
#include "include/node.h"
#include "include/packer.h"
#include "include/persisted_dictionary.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"
#include "include/program_context.h"
#include "include/ram_dictionary.h"
#include "include/shape.h"
#include "include/tensor_index.h"
#include "include/tensor_layout.h"
#include "include/waterline_rescale.h"
#include "test/test_constants.h"

namespace fhelipe {
class Cleartext;
}  // namespace fhelipe

std::shared_ptr<fhelipe::Node<fhelipe::TOp>> MakeInputNode(
    fhelipe::Dag<fhelipe::TOp>& top_dag, const fhelipe::TensorLayout& layout,
    const std::string& input_name);

void MakeOutputNode(fhelipe::Dag<fhelipe::TOp>& top_dag,
                    const std::shared_ptr<fhelipe::Node<fhelipe::TOp>>& parent,
                    const std::string& tensor_name);

fhelipe::Tensor<std::optional<fhelipe::PtVal>> ToOptionalTensor(
    const fhelipe::Tensor<fhelipe::PtVal>& tensor);

void DoChecks(
    const fhelipe::LaidOutTensorDictionary<fhelipe::PtChunk>& outputs,
    const fhelipe::Dictionary<fhelipe::Tensor<std::optional<fhelipe::PtVal>>>&
        check);

template <typename CtType>
fhelipe::IoManager<CtType> DummyIoManager(fhelipe::ChunkSize chunk_size) = delete;

fhelipe::RamDictionary<fhelipe::EncryptionConfig> GetEncryptionConfigs(
    const fhelipe::Dag<fhelipe::TOp>& top_dag);

template <typename CtType>
void AddEncryptedTensor(fhelipe::Dictionary<CtType>& ct_chunks,
                        const std::string& tensor_name,
                        const fhelipe::LaidOutTensor<CtType>& encrypted_tensor) {
  const auto& offsets = encrypted_tensor.Offsets();
  const auto& chunks = encrypted_tensor.Chunks();
  for (int idx = 0; idx < chunks.size(); ++idx) {
    ct_chunks.Record(
        fhelipe::ToFilename(fhelipe::IoSpec(tensor_name, offsets[idx].Flat())),
        chunks[idx].Chunk());
  }
}

inline std::vector<fhelipe::PtVal> IotaVector(int size) {
  std::vector<fhelipe::PtVal> result;
  for (int idx = 0; idx < size; ++idx) {
    result.push_back(idx);
  }
  return result;
}

fhelipe::RamDictionary<fhelipe::Shape> GatherFrontendTensors(
    const fhelipe::Dag<fhelipe::TOp>& top_dag);

template <class T>
void TestEqualVectors(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int idx = 0; idx < lhs.size(); ++idx) {
    ASSERT_EQ(lhs[idx], rhs[idx]);
  }
}

template <class T>
void TestEqualVectorsWithDontCares(const std::vector<T>& lhs,
                                   const std::vector<std::optional<T>>& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (int idx = 0; idx < lhs.size(); ++idx) {
    if (rhs[idx].has_value()) {
      ASSERT_EQ(lhs[idx], rhs[idx].value());
    }
  }
}

template <class CtType>
fhelipe::LaidOutTensor<CtType> EncryptTensor(
    const fhelipe::LaidOutTensor<fhelipe::PtChunk>& unencrypted_tensor) {
  auto context = fhelipe::ProgramContext{
      fhelipe::LogChunkSize(unencrypted_tensor.Chunks().at(0).Chunk().size()),
      kDefaultTestContext.LogScale(), kDefaultTestContext.UsableLevels(),
      kDefaultTestContext.GetBootstrappingPrecision()};
  return fhelipe::Convert<fhelipe::PtChunk, CtType>(
      unencrypted_tensor, [&context](const auto& chunk) {
        return Encrypt<CtType>(chunk, context);
      });
}

template <class CtType>
std::tuple<fhelipe::LaidOutTensor<CtType>, fhelipe::Tensor<fhelipe::PtVal>>
MakeInput(const fhelipe::TensorLayout& layout) {
  using fhelipe::PtVal;
  const fhelipe::Shape& shape = layout.GetShape();
  std::vector<PtVal> unencrypted(shape.ValueCnt());
  std::iota(unencrypted.begin(), unencrypted.end(), 0);
  fhelipe::LaidOutTensor<fhelipe::PtChunk> packed = Pack(unencrypted, layout);
  fhelipe::LaidOutTensor<CtType> encrypted = EncryptTensor<CtType>(packed);
  return std::make_tuple(encrypted,
                         fhelipe::Tensor(layout.GetShape(), unencrypted));
}

inline int UniformRandom(int low, int high) {
  CHECK(high > low);
  return low + std::rand() % (high - low);
}

inline std::vector<double> UniformRandomDouble(int sz, double low,
                                               double high) {
  CHECK(high > low);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(low, high);
  std::vector<double> result;
  while (result.size() < sz) {
    result.push_back(distribution(generator));
  }
  return result;
}

fhelipe::Shape RandomShape();
fhelipe::Shape RandomShape(int dimension_count);

fhelipe::TensorIndex RandomTensorIndex(const fhelipe::Shape& shape);
fhelipe::DiffTensorIndex RandomDiffTensorIndex(const fhelipe::Shape& shape);

fhelipe::ChunkSize RandomChunkSize();

fhelipe::TensorLayout RandomLayoutWithoutGaps(const fhelipe::Shape& shape,
                                             fhelipe::ChunkSize chunk_size);

fhelipe::TensorLayout RandomLayout();
fhelipe::TensorLayout RandomLayout(const fhelipe::Shape& shape);
fhelipe::TensorLayout RandomLayout(const fhelipe::Shape& shape,
                                  fhelipe::ChunkSize chunk_size);

fhelipe::LaidOutTensor<fhelipe::PtChunk> ToyTensor(
    const fhelipe::TensorLayout& layout);

std::tuple<fhelipe::LaidOutTensor<fhelipe::Cleartext>, std::vector<fhelipe::PtVal>>
MakeInput(const fhelipe::TensorLayout& layout);

void VerifyInvalidSlotsAreZero(const fhelipe::LaidOutTensor<fhelipe::PtChunk>& t);

fhelipe::RamDictionary<fhelipe::Tensor<fhelipe::PtVal>> MakeFrontendTensors(
    const fhelipe::Dictionary<fhelipe::Shape>& frontend_tensors);

template <class CtType>
std::pair<fhelipe::RamDictionary<CtType>,
          fhelipe::RamDictionary<fhelipe::Tensor<fhelipe::PtVal>>>
MakeInputs(const fhelipe::RamDictionary<fhelipe::EncryptionConfig>& configs) {
  fhelipe::RamDictionary<fhelipe::Tensor<fhelipe::PtVal>> tensor_dict;
  auto ct_chunks = fhelipe::RamDictionary<CtType>();
  fhelipe::ChunkSize chunk_size(1);
  for (const auto& tensor_name : configs.Keys()) {
    auto [lot, tensor] = MakeInput<CtType>(configs.At(tensor_name).Layout());
    AddEncryptedTensor(ct_chunks, tensor_name, lot);
    tensor_dict.Record(tensor_name, tensor);
    chunk_size = configs.At(tensor_name).Layout().ChunkSize().value();
  }

  auto context = fhelipe::ProgramContext(
      fhelipe::LogChunkSize(chunk_size), kDefaultTestContext.LogScale(),
      kDefaultTestContext.UsableLevels(),
      kDefaultTestContext.GetBootstrappingPrecision());
  // Add zero chunk
  ct_chunks.Record(
      ToFilename(fhelipe::IoSpec(fhelipe::kZeroCtName, 0)),
      Encrypt<CtType>(
          fhelipe::PtChunk(std::vector<fhelipe::PtVal>(chunk_size.value())),
          context));
  return {ct_chunks, tensor_dict};
}

template <class CtType>
fhelipe::LaidOutTensorDictionary<fhelipe::PtChunk> Compute(
    const fhelipe::ct_program::CtProgram& ct_program,
    const fhelipe::RamDictionary<fhelipe::EncryptionConfig>& configs,
    const fhelipe::Dictionary<CtType>& ct_inputs,
    const fhelipe::Dictionary<fhelipe::Tensor<fhelipe::PtVal>>& frontend_tensors) {
  auto io_manager = fhelipe::IoManager<CtType>(ct_inputs, frontend_tensors);
  auto result = fhelipe::Evaluator<CtType>::Evaluate(
      io_manager, ct_program, fhelipe::RamDictionary<CtType>());
  return fhelipe::Convert<fhelipe::PtChunk, CtType>(
      fhelipe::LaidOutTensorDictionary<CtType>(configs, *result),
      [](const CtType& ct) { return ct.Decrypt(); });
}

template <class CtType>
void DoTestIteration(
    const std::function<fhelipe::Dag<fhelipe::TOp>()>& make_top_dag,
    const std::function<
        fhelipe::RamDictionary<fhelipe::Tensor<std::optional<fhelipe::PtVal>>>(
            const fhelipe::Dictionary<fhelipe::Tensor<fhelipe::PtVal>>&)>& check) {
  auto top_dag = make_top_dag();
  auto configs = GetEncryptionConfigs(top_dag);
  auto inputs = MakeInputs<CtType>(configs);
  auto chunk_size = top_dag.NodesInTopologicalOrder()
                        .at(0)
                        ->Value()
                        .OutputLayout()
                        .ChunkSize();
  auto context = fhelipe::ProgramContext{
      fhelipe::LogChunkSize(chunk_size), kDefaultTestContext.LogScale(),
      kDefaultTestContext.UsableLevels(),
      kDefaultTestContext.GetBootstrappingPrecision()};
  // TODO(nsamar): Turn this into a call to the Compiler
  auto ct_program =
      fhelipe::BasicCtOpPass(
          context, std::make_unique<fhelipe::RamDictionary<fhelipe::ChunkIr>>(
                       fhelipe::RamDictionary<fhelipe::ChunkIr>()))
          .DoPass(fhelipe::LazyBootstrappingPass(context).DoPass(
              fhelipe::WaterlineRescale(context).DoPass(top_dag)));
  const auto& frontend_tensors =
      MakeFrontendTensors(GatherFrontendTensors(top_dag));

  auto outputs =
      Compute<CtType>(ct_program, configs, inputs.first, frontend_tensors);
  fhelipe::AppendDictionary(inputs.second, frontend_tensors);
  DoChecks(outputs, check(inputs.second));
}

template <class CtType>
void DoTest(
    const std::function<fhelipe::Dag<fhelipe::TOp>()>& make_top_dag,
    const std::function<
        fhelipe::RamDictionary<fhelipe::Tensor<std::optional<fhelipe::PtVal>>>(
            const fhelipe::Dictionary<fhelipe::Tensor<fhelipe::PtVal>>&)>& check) {
  for (int idx : Estd::indices(kIterationsPerTest)) {
    (void)idx;
    DoTestIteration<CtType>(make_top_dag, check);
  }
}

#endif  // FHELIPE_TEST_UTILS_H_
