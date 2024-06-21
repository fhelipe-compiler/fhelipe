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

#ifndef FHELIPE_FHEBOOSTER_PASS_H_
#define FHELIPE_FHEBOOSTER_PASS_H_

#include "pass.h"
#include "pass_utils.h"

namespace fhelipe {

class FheBoosterPass : public CtOpOptimizer {
 public:
     FheBoosterPass(const Level& usable_levels) : usable_levels_(usable_levels) {}
     CtOpOptimizerOutput DoPass(const CtOpOptimizerInput& in_dag) final;

     const PassName& GetPassName() const final {
         static PassName pass_name("fhe_booster_pass");
         return pass_name;
     }

     std::unique_ptr<CtOpOptimizer> CloneUniq() const final {
         return std::make_unique<FheBoosterPass>(usable_levels_);
     }

 private:
     Level usable_levels_;
};

}

#endif  // FHELIPE_FHEBOOSTER_PASS_H_
