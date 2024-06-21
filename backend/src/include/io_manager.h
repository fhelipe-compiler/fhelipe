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

#ifndef IO_MANAGER_H_
#define IO_MANAGER_H_

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "dictionary.h"
#include "encryption_config.h"
#include "filesystem_utils.h"
#include "io_spec.h"
#include "tensor.h"
#include "utils.h"

namespace fhelipe {

template <class CtType>
class IoManager {
 public:
  explicit IoManager(const Dictionary<CtType>& ct_chunks,
                     const Dictionary<Tensor<PtVal>>& frontend_tensors)
      : ct_chunks_(ct_chunks.CloneUniq()),
        frontend_tensors_(frontend_tensors.CloneUniq()) {}
  IoManager() : frontend_tensors_(nullptr) {}
  IoManager(IoManager&&) = default;
  virtual ~IoManager() {}
  IoManager(const IoManager&) = delete;
  IoManager& operator=(const IoManager&) = delete;
  IoManager& operator=(IoManager&&) = default;

  CtType Ct(const IoSpec& spec) const {
    return ct_chunks_->At(ToFilename(spec));
  }

  Tensor<PtVal> T(const std::string& name) const {
    return frontend_tensors_->At(name);
  }

  const Dictionary<Tensor<PtVal>>& FrontendTensors() const {
    return *frontend_tensors_;
  }

 private:
  std::unique_ptr<Dictionary<CtType>> ct_chunks_;
  std::unique_ptr<Dictionary<Tensor<PtVal>>> frontend_tensors_;
};

}  // namespace fhelipe

#endif  // IO_MANAGER_H_
