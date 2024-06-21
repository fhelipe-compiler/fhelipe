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

#ifndef FHELIPE_DUMMY_CT_OP_PASS_H_
#define FHELIPE_DUMMY_CT_OP_PASS_H_

#include <memory>

#include "pass.h"
#include "pass_utils.h"

namespace fhelipe {

// DummyCtOpPass just returns an empty CtProgram.
// Used to improve compile time when you only care about passes before CtOpPass
// (the CtOpPass is always the performance bottleneck, because it needs to
// create many CtOps).
class DummyCtOpPass : public CtOpPass {
 public:
  DummyCtOpPass(const ProgramContext& context,
                std::unique_ptr<Dictionary<ChunkIr>>&& chunk_dict)
      : context_(context), chunk_dict_(std::move(chunk_dict)) {}
  const PassName& GetPassName() const final {
    static PassName pass_name("dummy_ct_op_pass");
    return pass_name;
  }

  std::unique_ptr<CtOpPass> CloneUniq() const final {
    return std::make_unique<DummyCtOpPass>(context_, chunk_dict_->CloneUniq());
  }

  CtOpPassOutput DoPass(const CtOpPassInput& in_dag) final {
    (void)in_dag;
    return {context_, *chunk_dict_};
  }

 private:
  ProgramContext context_;
  std::unique_ptr<Dictionary<ChunkIr>> chunk_dict_;
};

}  // namespace fhelipe

#endif  // FHELIPE_DUMMY_CT_OP_PASS_H_
