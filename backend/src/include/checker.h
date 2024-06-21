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

#ifndef FHELIPE_CHECKER_H_
#define FHELIPE_CHECKER_H_

#include <vector>

#include "plaintext.h"

namespace fhelipe {

void TestCloseEnough(const std::vector<fhelipe::PtVal>& v0,
                     const std::vector<fhelipe::PtVal>& v1);
void TestCloseEnough(const std::vector<fhelipe::PtVal>& v0,
                     const std::vector<fhelipe::PtVal>& v1, double epsilon);

fhelipe::PtVal LInfinityNorm(const std::vector<fhelipe::PtVal>& values);
fhelipe::PtVal LInfinityDiff(const std::vector<fhelipe::PtVal>& chunk0,
                            const std::vector<fhelipe::PtVal>& chunk1);

void CheckSmallEnoughForBootstrapping(const std::vector<fhelipe::PtVal>& values);

}  // namespace fhelipe

#endif  // FHELIPE_CHECKER_H_
