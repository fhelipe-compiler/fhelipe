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

#ifndef MAYBE_TENSOR_INDEX_H_
#define MAYBE_TENSOR_INDEX_H_

#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <ostream>

#include "constants.h"
#include "extended_std.h"
#include "io_utils.h"
#include "tensor_index.h"

namespace fhelipe {

using MaybeTensorIndex = std::optional<TensorIndex>;

inline std::vector<std::optional<int>> FlatMaybeTensorIndices(
    const std::vector<MaybeTensorIndex>& mti_vector) {
  auto result =
      Estd::transform(mti_vector, [](const auto& ti) -> std::optional<int> {
        if (ti.has_value()) {
          return ti.value().Flat();
        }
        return std::nullopt;
      });
  return result;
}

}  // namespace fhelipe

#endif  // MAYBE_TENSOR_INDEX_H_
