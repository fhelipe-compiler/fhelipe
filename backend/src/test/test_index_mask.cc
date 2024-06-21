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

#include <memory>

#include "gtest/gtest.h"

#include "include/index_mask.h"
#include "include/plaintext.h"
#include "test/test_utils.h"

using namespace fhelipe;

/*
 *
TEST(IndexMask, MaskedIndices) {
  int test_index = 100;
  TestEqualVectors(MaskedIndices(IndexMask(test_index)), {2, 5, 6});
}

TEST(IndexMask, MaxIndexMask) { ASSERT_EQ(MaxIndexMask(5).to_ulong(), 8 - 1); }
*/
