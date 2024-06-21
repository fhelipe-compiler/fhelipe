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

#ifndef FHELIPE_ENCRYPTOR_H_
#define FHELIPE_ENCRYPTOR_H_

#include <latticpp/latticpp.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "constants.h"
#include "encryption_config.h"
#include "extended_std.h"
#include "filesystem_utils.h"
#include "include/dictionary.h"
#include "io_spec.h"
#include "io_utils.h"
#include "laid_out_tensor.h"
#include "laid_out_tensor_utils.h"
#include "latticpp/ckks/lattigo_param.h"
#include "plaintext.h"
#include "plaintext_chunk.h"
#include "tensor.h"
#include "tensor_index.h"
#include "tensor_layout.h"
#include "utils.h"

namespace latticpp {
class LattigoParam;
}  // namespace latticpp

namespace fhelipe {

template <class T>
LaidOutTensor<PtChunk> Decrypt(const LaidOutTensor<T>& tensor_c) {
  return Convert<T, PtChunk>(tensor_c,
                             [](const auto& ct) { return ct.Decrypt(); });
}

Tensor<PtVal> Unpack(const LaidOutTensor<PtChunk>& tensor);

LaidOutTensor<PtChunk> Pack(const std::vector<PtVal>& vec,
                            const TensorLayout& layout);

inline LaidOutTensor<PtChunk> Pack(const Tensor<PtVal>& tensor,
                                   const TensorLayout& layout) {
  return Pack(tensor.Values(), layout);
}

}  // namespace fhelipe

#endif  // FHELIPE_ENCRYPTOR_H_
