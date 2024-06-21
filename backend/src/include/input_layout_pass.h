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

#ifndef FHELIPE_INPUT_LAYOUT_PASS_H_
#define FHELIPE_INPUT_LAYOUT_PASS_H_

#include "pass.h"
#include "pass_utils.h"

namespace fhelipe {

class InputLayoutPass : public LayoutOptimizer {
 public:
  InputLayoutPass() = default;
  LayoutOptimizerOutput DoPass(const LayoutOptimizerInput& in_dag) final;

  const PassName& GetPassName() const final {
    static PassName pass_name("input_layout_pass");
    return pass_name;
  }

  std::unique_ptr<LayoutOptimizer> CloneUniq() const final {
    return std::make_unique<InputLayoutPass>();
  }

 private:
};

}  // namespace fhelipe

#endif  // FHELIPE_INPUT_LAYOUT_PASS_H_
