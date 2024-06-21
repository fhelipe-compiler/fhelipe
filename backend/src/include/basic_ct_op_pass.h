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

#ifndef FHELIPE_BASIC_CT_OP_PASS_H_
#define FHELIPE_BASIC_CT_OP_PASS_H_

#include <memory>
#include <string>

#include "chunk_ir.h"
#include "dag.h"
#include "debug_info.h"
#include "dictionary.h"
#include "include/pass_utils.h"
#include "leveled_t_op.h"
#include "program_context.h"

namespace fhelipe {

template <class T>
class Dag;

namespace ct_program {
class CtProgram;
}

class CtOp;
class TOp;

class BasicCtOpPass : public CtOpPass {
 public:
  BasicCtOpPass(const ProgramContext& context,
                std::unique_ptr<Dictionary<ChunkIr>>&& chunk_dict);
  CtOpPassOutput DoPass(const CtOpPassInput& in_dag) final;
  const PassName& GetPassName() const final {
    static PassName pass_name("basic_ct_op_pass");
    return pass_name;
  }
  std::unique_ptr<CtOpPass> CloneUniq() const final {
    return std::make_unique<BasicCtOpPass>(context_, chunk_dict_->CloneUniq());
  }

 private:
  ProgramContext context_;
  std::unique_ptr<Dictionary<ChunkIr>> chunk_dict_;
};

}  // namespace fhelipe

#endif  // FHELIPE_BASIC_CT_OP_PASS_H_
