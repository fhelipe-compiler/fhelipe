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

#ifndef FHELIPE_LAID_OUT_TENSOR_DICTIONARY_H_
#define FHELIPE_LAID_OUT_TENSOR_DICTIONARY_H_

#include <algorithm>
#include <memory>

#include "dictionary.h"
#include "encryption_config.h"
#include "include/chunk_ir.h"
#include "include/extended_std.h"
#include "include/io_spec.h"
#include "include/laid_out_tensor.h"
#include "include/packer.h"
#include "include/program_context.h"
#include "tensor.h"

namespace fhelipe {

// This class encapsulates the implementation detail that `LaidOutTensor`s are
// usually stored in such a way where the actual chunk data and chunk layout
// information are separated. This separation is useful because different parts
// of Fhelipe need/have access to only one of these. For example:
//
// 1) The compiler will generate the layout info, but will not be aware of the
// data.
//
// 2) The runtime requires only input data and produces only output data,
// without being aware of the layout.
//
// 3) The encryptor and decryptor need to be aware of both layout and data.
//
// Thus, this class is useful for the encryptor and decryptor, because these
// two components need both layout and data info together.
template <typename ChunkType>
class LaidOutTensorDictionary {
 public:
  LaidOutTensorDictionary(const Dictionary<EncryptionConfig>& configs,
                          const Dictionary<ChunkType>& chunks)
      : configs_(configs.CloneUniq()), chunks_(chunks.CloneUniq()) {
    // Ensure configs and chunks are mutually consistent
    for (const auto& chunk_name : chunks_->Keys()) {
      auto io_spec = FilenameToIoSpec(chunk_name);
      CHECK(Estd::contains(configs_->Keys(), io_spec.name));
      auto tensor_layout = configs_->At(io_spec.name).Layout();
      CHECK(
          Estd::contains(tensor_layout.ChunkOffsets(),
                         TensorIndex(tensor_layout.GetShape(), io_spec.offset)))
          << io_spec.name << " " << io_spec.offset;
    }
  }

  const Dictionary<EncryptionConfig>& Config() const { return *configs_; }

  LaidOutTensor<ChunkType> At(const std::string& tensor_name) const;

  LaidOutChunk<ChunkType> At(const IoSpec& io_spec) const;
  const Dictionary<ChunkType>& Chunks() const { return *chunks_; }

  void Record(const IoSpec& io_spec, const ChunkType& chunk);

 private:
  std::unique_ptr<Dictionary<EncryptionConfig>> configs_;
  std::unique_ptr<Dictionary<ChunkType>> chunks_;
};

template <typename ChunkType>
void LaidOutTensorDictionary<ChunkType>::Record(const IoSpec& io_spec,
                                                const ChunkType& chunk) {
  CHECK(io_spec.name == kZeroCtName ||
        Estd::contains(configs_->Keys(), io_spec.name))
      << io_spec.name << " " << io_spec.offset;
  chunks_->Record(ToFilename(io_spec), chunk);
}

template <class ChunkType>
void Record(LaidOutTensorDictionary<ChunkType>& dict,
            const std::string& tensor_name,
            const LaidOutTensor<ChunkType>& lot) {
  Estd::for_each(lot.Offsets(), lot.Chunks(),
                 [&dict, &tensor_name](const auto& offset, const auto& chunk) {
                   return dict.Record(IoSpec(tensor_name, offset.Flat()),
                                      chunk.Chunk());
                 });
}

template <class OutType, class InType>
LaidOutTensorDictionary<OutType> Convert(
    const LaidOutTensorDictionary<InType>& in_dict,
    const std::function<OutType(InType)>& func) {
  return {in_dict.Config(), Convert<OutType>(in_dict.Chunks(), func)};
}

template <class ChunkType>
LaidOutChunk<ChunkType> LaidOutTensorDictionary<ChunkType>::At(
    const IoSpec& io_spec) const {
  const auto& layout = configs_->At(io_spec.name).Layout();
  const auto& offset = TensorIndex(layout.GetShape(), io_spec.offset);
  return {layout, offset, chunks_->At(ToFilename(io_spec))};
}

template <class ChunkType>
LaidOutTensor<ChunkType> LaidOutTensorDictionary<ChunkType>::At(
    const std::string& tensor_name) const {
  EncryptionConfig tensor_config = configs_->At(tensor_name);
  const auto& layout = tensor_config.Layout();
  std::vector<LaidOutChunk<ChunkType>> chunks = Estd::transform(
      tensor_config.Layout().ChunkOffsets(),
      [this, &tensor_name, &layout](const auto& offset) {
        return LaidOutChunk<ChunkType>(
            layout, offset,
            chunks_->At(ToFilename(IoSpec(tensor_name, offset.Flat()))));
      });
  return LaidOutTensor<ChunkType>{chunks};
}

void MakeUnencDictionaryFromTensors(
    LaidOutTensorDictionary<PtChunk>& unenc_dict,
    const Dictionary<Tensor<PtVal>>& tensor_dict);

template <typename CtType>
void Encrypt(Dictionary<CtType>& enc_dict,
             const Dictionary<PtChunk>& unenc_dict,
             const ProgramContext& context) {
  for (const auto& key : unenc_dict.Keys()) {
    enc_dict.Record(key, Encrypt<CtType>(unenc_dict.At(key), context));
  }
}

template <typename CtType>
void Decrypt(Dictionary<PtChunk>& unenc_dict,
             const Dictionary<CtType>& enc_dict) {
  for (const auto& key : enc_dict.Keys()) {
    std::cout << key << std::endl;
    unenc_dict.Record(key, enc_dict.At(key).Decrypt());
  }
}

}  // namespace fhelipe

#endif  // FHELIPE_LAID_OUT_TENSOR_DICTIONARY_H_
