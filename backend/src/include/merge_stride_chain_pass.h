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

#ifndef FHELIPE_MERGE_STRIDE_CHAIN_PASS_H_
#define FHELIPE_MERGE_STRIDE_CHAIN_PASS_H_

#include "pass.h"
#include "pass_utils.h"
#include "utils.h"

namespace fhelipe {

class MergeStrideChainPass : public EmbrioOptimizer {
 public:
  MergeStrideChainPass() {}

  EmbrioOptimizerOutput DoPass(const EmbrioOptimizerOutput& in_dag) final;

  const PassName& GetPassName() const final {
    static PassName pass_name("merge_stride_chain_pass");
    return pass_name;
  }

  std::unique_ptr<EmbrioOptimizer> CloneUniq() const final {
    return std::make_unique<MergeStrideChainPass>();
  }

 private:
};

}  // namespace fhelipe

#endif  // FHELIPE_MERGE_STRIDE_CHAIN_PASS_H_
