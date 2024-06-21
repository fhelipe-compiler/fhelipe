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

#ifndef FHELIPE_BASIC_PREPROCESSOR_H_
#define FHELIPE_BASIC_PREPROCESSOR_H_

#include <filesystem>

#include "pass.h"
#include "pass_utils.h"

namespace fhelipe {

class LogScale;
class Level;

class BasicPreprocessor : public Preprocessor {
 public:
  BasicPreprocessor(LogScale default_log_scale, Level max_usable_level)
      : default_log_scale_(default_log_scale),
        max_usable_level_(max_usable_level) {}

  PreprocessorOutput DoPass(const PreprocessorInput& exe_folder) final;
  const PassName& GetPassName() const final {
    static auto pass_name = PassName("basic_preprocessor");
    return pass_name;
  }
  std::unique_ptr<Preprocessor> CloneUniq() const final {
    return std::make_unique<BasicPreprocessor>(*this);
  }

 private:
  LogScale default_log_scale_;
  Level max_usable_level_;
};

}  // namespace fhelipe

#endif  // FHELIPE_BASIC_PREPROCESSOR_H_
