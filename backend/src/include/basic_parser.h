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

#ifndef FHELIPE_BASIC_PARSER_H_
#define FHELIPE_BASIC_PARSER_H_

#include "pass_utils.h"

namespace fhelipe {

class BasicParser : public Parser {
 public:
  ParserOutput DoPass(const ParserInput& code) final;

  const PassName& GetPassName() const final {
    static PassName pass_name = PassName("basic_parser");
    return pass_name;
  }

  std::unique_ptr<Parser> CloneUniq() const final {
    return std::make_unique<BasicParser>(*this);
  }

 private:
};

}  // namespace fhelipe

#endif  // FHELIPE_BASIC_PARSER_H_
