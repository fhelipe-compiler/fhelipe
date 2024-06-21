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

#ifndef FHELIPE_CT_OP_VISITOR_H_
#define FHELIPE_CT_OP_VISITOR_H_

#include "node.h"

namespace fhelipe {

class CtOp;
class AddCC;
class AddCP;
class AddCS;
class MulCC;
class MulCP;
class MulCS;
class RotateC;
class InputC;
class RescaleC;
class OutputC;
class BootstrapC;

class CtOpVisitor {
 public:
  CtOpVisitor() = default;
  virtual ~CtOpVisitor() {}
  CtOpVisitor(const CtOpVisitor&) = delete;
  CtOpVisitor(CtOpVisitor&&) = default;
  CtOpVisitor& operator=(const CtOpVisitor&) = delete;

  virtual void Visit(const Node<CtOp>& node) = 0;
};

}  // namespace fhelipe

#endif  // FHELIPE_CT_OP_VISITOR_H_
