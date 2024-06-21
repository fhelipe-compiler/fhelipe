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

#ifndef FHELIPE_LAZY_BOOTSTRAPPING_PASS_H_
#define FHELIPE_LAZY_BOOTSTRAPPING_PASS_H_

#include <unordered_map>
#include <vector>

#include "include/pass_utils.h"
#include "program_context.h"

namespace latticpp {
class LattigoParam;
}  // namespace latticpp

namespace fhelipe {

template <class T>
class Dag;
class TOp;
class TOpEmbrio;
class ScaledTOp;
class LeveledTOp;

class LazyBootstrappingPass : public LevelingPass {
 public:
  explicit LazyBootstrappingPass(const ProgramContext& context)
      : context_(context) {}
  LevelingPassOutput DoPass(const LevelingPassInput& in_dag) final;
  const PassName& GetPassName() const final {
    static PassName pass_name("lazy_bootstrapping_pass");
    return pass_name;
  }
  std::unique_ptr<LevelingPass> CloneUniq() const final {
    return std::make_unique<LazyBootstrappingPass>(context_);
  }

 private:
  ProgramContext context_;
};

}  // namespace fhelipe

#endif  // FHELIPE_LAZY_BOOTSTRAPPING_PASS_H_
