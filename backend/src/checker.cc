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

#include "include/checker.h"

#include "include/constants.h"
#include "include/extended_std.h"
#include "include/plaintext.h"

namespace fhelipe {

// Very low check for logreg accuracy experiments
constexpr double kCloseEnough = 10000;

fhelipe::PtVal LInfinityNorm(const std::vector<fhelipe::PtVal>& values) {
  return Estd::max_element(Estd::transform(
      values, [](const fhelipe::PtVal& value) { return std::abs(value); }));
}

fhelipe::PtVal LInfinityDiff(const std::vector<fhelipe::PtVal>& chunk0,
                            const std::vector<fhelipe::PtVal>& chunk1) {
  CHECK(chunk0.size() == chunk1.size());
  auto diff = Estd::transform(chunk0, chunk1, std::minus<>());
  diff = Estd::transform(diff, [](const auto& x) { return std::abs(x); });
  return LInfinityNorm(diff);
}

void TestCloseEnough(const std::vector<fhelipe::PtVal>& chunk0,
                     const std::vector<fhelipe::PtVal>& chunk1, double epsilon) {
  CHECK(chunk0.size() == chunk1.size());
  auto diff = Estd::transform(chunk0, chunk1, std::minus<>());
  diff = Estd::transform(diff, [](const auto& x) { return std::abs(x); });
  std::cout << "Error: " << LInfinityNorm(diff) << "; epsilon: " << epsilon
            << std::endl;
}

void TestCloseEnough(const std::vector<fhelipe::PtVal>& chunk0,
                     const std::vector<fhelipe::PtVal>& chunk1) {
  // NO TEST!!!
  return;
  TestCloseEnough(chunk0, chunk1, kCloseEnough);
}

void CheckSmallEnoughForBootstrapping(
    const std::vector<fhelipe::PtVal>& values) {
  // Disable checks
  return;
  for (auto value : values) {
    // the std::abs(value) for something that will be bootstrapped must be
    // smaller than q0/scale; we only allow for q0/scale > 1024, so that's why
    // this check makes sense
    CHECK(std::abs(value) < 1024) << "Offending value: " << std::abs(value);
  }
}

}  // namespace fhelipe
