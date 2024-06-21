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

#include "include/glue_code.h"

#include <filesystem>

#include "include/constants.h"
#include "include/dag_io.h"
#include "include/encryption_config.h"
#include "include/io_spec.h"
#include "include/io_utils.h"
#include "include/laid_out_tensor_dictionary.h"
#include "include/persisted_dictionary.h"
#include "include/ram_dictionary.h"
#include "include/t_op_embrio.h"
#include "latticpp/ckks/lattigo_param.h"

namespace fhelipe {

LaidOutTensorDictionary<PtChunk> CreateUnencryptedInputs(
    const std::filesystem::path& folder_path) {
  auto tensor_dict_path = folder_path / kInUnenc;
  auto tensor_dict = PersistedDictionary<Tensor<PtVal>>(tensor_dict_path);
  CHECK(!tensor_dict.Keys().empty()) << tensor_dict_path;
  auto config_path = folder_path / kEncCfg;
  auto config = PersistedDictionary<EncryptionConfig>(config_path);
  CHECK(!config.Keys().empty()) << config_path;
  auto unenc_chunks = RamDictionary<PtChunk>();
  auto unenc_dict = LaidOutTensorDictionary<PtChunk>(config, unenc_chunks);
  MakeUnencDictionaryFromTensors(unenc_dict, tensor_dict);
  // Add chunk of zeros
  unenc_dict.Record(
      IoSpec(kZeroCtName, 0),
      PtChunk(std::vector<PtVal>(
          config.At(*config.Keys().begin()).Layout().ChunkSize().value(), 0)));
  return unenc_dict;
}

}  // namespace fhelipe
