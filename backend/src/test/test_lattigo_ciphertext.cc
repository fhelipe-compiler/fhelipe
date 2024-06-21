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

/*
#include <iostream>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "include/chunk.h"
#include "include/lattigo_ct.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"
#include "latticpp/ckks/lattigo_param.h"
#include "test/test_utils.h"

using namespace fhelipe;

latticpp::LattigoParam param(16, 45, 10, 32);
int slot_count = 1 << param.LogScale();

TEST(LattigoCt, AddCC) {
  PtChunk cmsg0(IotaVector(slot_count));
  PtChunk cmsg1(IotaVector(slot_count));
  LattigoCt ct0 = Encrypt<LattigoCt>(cmsg0, param);
  LattigoCt ct1 = Encrypt<LattigoCt>(cmsg1, param);
  LattigoCt result = ct0.AddCC(ct1);
  TestCloseEnough(result.Decrypt().Values(), Add(cmsg0, cmsg1).Values());
}

TEST(LattigoCt, AddCP) {
  PtChunk cmsg0(IotaVector(slot_count));
  PtChunk cmsg1(IotaVector(slot_count));
  LattigoCt ct0 = Encrypt<LattigoCt>(cmsg0, param);
  LattigoCt result = ct0.AddCP(cmsg1);
  TestCloseEnough(result.Decrypt().Values(), Add(cmsg0, cmsg1).Values());
}

TEST(LattigoCt, BootstrapC) {
  PtChunk cmsg0(UniformRandomDouble(slot_count, -1.0, 1.0));
  LattigoCt ct0 = Encrypt<LattigoCt>(cmsg0, param);
  TestCloseEnough(ct0.Decrypt().Values(), cmsg0.Values());
  auto result = ct0.BootstrapC();
  TestCloseEnough(ct0.Decrypt().Values(), result.Decrypt().Values());
}

TEST(LattigoCt, Conv) {
  PtChunk in(IotaVector(slot_count));
  PtChunk zeros(std::vector<PtVal>(slot_count, 0));
  PtChunk pt_mask(std::vector<PtVal>(slot_count, 1));
  LattigoCt ct = Encrypt<LattigoCt>(in, param);
  LattigoCt ct_zeros = Encrypt<LattigoCt>(zeros, param);
  auto masked = ct.MulCP(pt_mask);
  auto rotated = masked.RotateC(0);
  auto added = rotated.AddCC(ct_zeros);
  TestCloseEnough(ct.Decrypt().Values(), IotaVector(slot_count));
  std::cout << "ct ok" << std::endl;
  TestCloseEnough(masked.Decrypt().Values(), IotaVector(slot_count));
  std::cout << "masked ok" << std::endl;
  TestCloseEnough(rotated.Decrypt().Values(), IotaVector(slot_count));
  std::cout << "rotated ok" << std::endl;
  TestCloseEnough(added.Decrypt().Values(), IotaVector(slot_count));
  std::cout << "added ok" << std::endl;
}

TEST(LattigoCt, MulCC) {
  PtChunk cmsg0(IotaVector(slot_count));
  PtChunk cmsg1(IotaVector(slot_count));
  LattigoCt ct0 = Encrypt<LattigoCt>(cmsg0, param);
  LattigoCt ct1 = Encrypt<LattigoCt>(cmsg1, param);
  LattigoCt result = ct0.MulCC(ct1);
  TestCloseEnough(result.Decrypt().Values(), Mul(cmsg0, cmsg1).Values());
}

TEST(LattigoCt, MulCP) {
  PtChunk cmsg0(IotaVector(slot_count));
  PtChunk cmsg1(IotaVector(slot_count));
  LattigoCt ct0 = Encrypt<LattigoCt>(cmsg0, param);
  LattigoCt result = ct0.MulCP(cmsg1);
  TestCloseEnough(result.Decrypt().Values(), Mul(cmsg0, cmsg1).Values());
}

TEST(LattigoCt, RotateC) {
  PtChunk cmsg0(IotaVector(slot_count));

  std::vector<int> rotate_bys(17);
  std::iota(rotate_bys.begin(), rotate_bys.end(), -8);

  for (int rotate_by : rotate_bys) {
    LattigoCt ct0 = Encrypt<LattigoCt>(cmsg0, param);
    LattigoCt result = ct0.RotateC(rotate_by);
    auto res = result.Decrypt().Values();
    auto check = Rotate(cmsg0, rotate_by).Values();
    TestCloseEnough(res, check);
  }
}
*/
