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

#ifndef FHELIPE_TEST_CONSTANTS_H_
#define FHELIPE_TEST_CONSTANTS_H_

#include "include/constants.h"
#include "include/program_context.h"

constexpr int kIterationsPerTest = 100;
constexpr int kLogMaxSlotSize = 7;
constexpr int kMaxTensorElements = 1 << 7;
constexpr int kMaxDimensionCount = 5;
constexpr int kMaxTestLayoutGaps = 5;
constexpr int kMaxTestReplicationAmount = 10;
constexpr int kMaxLogTestStrideSize = 6;

static const fhelipe::ProgramContext kDefaultTestContext{
    fhelipe::kDefaultLogChunkSize, fhelipe::kDefaultLogScale,
    fhelipe::kDefaultUsableLevels, fhelipe::kDefaultBootstrappingPrecision};

#endif  // FHELIPE_TEST_CONSTANTS_H_
