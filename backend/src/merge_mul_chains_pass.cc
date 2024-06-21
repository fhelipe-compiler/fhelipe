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

#include "include/merge_mul_chains_pass.h"

#include "include/pass_utils.h"
#include "include/t_merged_mul_chain_cp.h"

namespace fhelipe {

// nsamar: Does nothing currently
LayoutOptimizerOutput MergeMulChainsPass::DoPass(
    const LayoutOptimizerInput& in_dag) {
  Dag<TOp> out_dag = CloneFromAncestor(in_dag);
  return out_dag;
}

}  // namespace fhelipe
