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

#ifndef FHELIPE_GLUE_CODE_H_
#define FHELIPE_GLUE_CODE_H_

#include <filesystem>

#include "ct_program.h"
#include "dictionary.h"
#include "evaluator.h"
#include "include/constants.h"
#include "include/program_context.h"
#include "io_manager.h"
#include "laid_out_tensor_dictionary.h"
#include "persisted_dictionary.h"
#include "ram_dictionary.h"

namespace fhelipe {

LaidOutTensorDictionary<PtChunk> CreateUnencryptedInputs(
    const std::filesystem::path& folder_path);

inline void Unpack(Dictionary<Tensor<PtVal>>& tensor_dict,
                   const LaidOutTensorDictionary<PtChunk>& chunk_dict) {
  for (const KeyType& tensor_name : chunk_dict.Config().Keys()) {
    if (!chunk_dict.Config().At(tensor_name).IsInput()) {
      const LaidOutTensor<PtChunk>& lot = chunk_dict.At(tensor_name);
      tensor_dict.Record(tensor_name, Unpack(lot));
    }
  }
}

template <class CtType>
void Decrypt(const std::filesystem::path& folder_path) {
  auto config = PersistedDictionary<EncryptionConfig>(folder_path / kEncCfg);
  auto enc_dict = PersistedDictionary<CtType>(folder_path / kOutEnc);
  std::cout << "out_enc: " << folder_path / kOutEnc << std::endl;
  auto unenc_dict = RamDictionary<PtChunk>();
  Decrypt<CtType>(unenc_dict, enc_dict);
  auto results_dict =
      ClearedPersistedDictionary<Tensor<PtVal>>(folder_path / kOutUnenc);
  Unpack(results_dict, LaidOutTensorDictionary<PtChunk>(config, unenc_dict));
}

template <class CtType>
void Run(const std::filesystem::path& folder_path) {
  auto stream = OpenStream<std::ifstream>(folder_path / kExecutable);
  auto ct_program = ReadStream<ct_program::CtProgram>(stream);
  auto ct_chunks = PersistedDictionary<CtType>(folder_path / kInEnc);
  auto frontend_tensors =
      PersistedDictionary<Tensor<PtVal>>(folder_path / kPts);
  IoManager<CtType> io_manager(ct_chunks, frontend_tensors);
  auto outputs = ClearedPersistedDictionary<CtType>(folder_path / kOutEnc);
  auto result = Evaluator<CtType>::Evaluate(io_manager, ct_program, outputs);
}

template <class CtType>
void Encrypt(const std::filesystem::path& folder_path,
             const ProgramContext& context) {
  auto enc_dict = ClearedPersistedDictionary<CtType>(folder_path / kInEnc);
  auto unenc_dict = CreateUnencryptedInputs(folder_path);
  Encrypt<CtType>(enc_dict, unenc_dict.Chunks(), context);
}

}  // namespace fhelipe

#endif  // FHELIPE_GLUE_CODE_H_
