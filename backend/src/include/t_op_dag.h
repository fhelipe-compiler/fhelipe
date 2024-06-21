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

#ifndef FHELIPE_T_OP_DAG_H_
#define FHELIPE_T_OP_DAG_H_

#include "dag.h"
#include "t_op.h"

namespace fhelipe {

class TOpDag {
 public:
  TOpDag();
  TOpDag(TOpDag&&) = default;
  virtual ~TOpDag() {}
  TOpDag(const TOpDag&) = delete;
  TOpDag& operator=(const TOpDag&) = delete;
  TOpDag& operator=(TOpDag&&) = default;

  const TOp& AddNode(std::unique_ptr<TOp> new_node,
                     const std::vector<const TOp*>& parents);

  const std::vector<const TOp*>& Parents(const TOp& node) const;
  const std::vector<const TOp*>& Children(const TOp& node) const;

  std::vector<const TOp*> Nodes() const;
  bool Contains(const TOp& node) const;

 private:
  Dag<TOp> ct_op_dag_;
};

}  // namespace fhelipe

#endif  // FHELIPE_T_OP_DAG_H_
