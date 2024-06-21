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

#ifndef FHELIPE_WATERLINE_RESCALE_H_
#define FHELIPE_WATERLINE_RESCALE_H_

#include "debug_info.h"
#include "pass_utils.h"
#include "program_context.h"
#include "scaled_t_op.h"

namespace fhelipe {

class WaterlineRescale : public RescalingPass {
 public:
  explicit WaterlineRescale(const ProgramContext& context)
      : context_(context) {}
  RescalingPassOutput DoPass(const RescalingPassInput& in_dag) final;
  const PassName& GetPassName() const final {
    static PassName pass_name = PassName("waterline_rescale");
    return pass_name;
  }
  std::unique_ptr<RescalingPass> CloneUniq() const final {
    return std::make_unique<WaterlineRescale>(*this);
  }

 private:
  ProgramContext context_;
};

}  // namespace fhelipe

#endif  // FHELIPE_WATERLINE_RESCALE_H_
