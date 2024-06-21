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

#include "include/laid_out_tensor_dictionary.h"

#include "include/laid_out_tensor.h"
#include "include/packer.h"

namespace fhelipe {

void MakeUnencDictionaryFromTensors(
    LaidOutTensorDictionary<PtChunk>& unenc_dict,
    const Dictionary<Tensor<PtVal>>& tensor_dict) {
  Estd::for_each(tensor_dict.Keys(),
                 [&unenc_dict, &tensor_dict](const auto& tensor_name) {
                   Record(unenc_dict, tensor_name,
                          Pack(tensor_dict.At(tensor_name),
                               unenc_dict.Config().At(tensor_name).Layout()));
                 });
}

}  // namespace fhelipe
