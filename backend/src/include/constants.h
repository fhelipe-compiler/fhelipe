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

#ifndef FHELIPE_CONSTANTS_H_
#define FHELIPE_CONSTANTS_H_

#include <string>

#include "include/bootstrapping_precision.h"
#include "include/chunk_size.h"

namespace fhelipe {

static const std::string kChIr = "ch_ir";
static const std::string kExecutable = "rt.df";
static const std::string kSchedulableDfg = "rt.axel";
static const std::string kDslOutput = "t.df";
static const std::string kDslPreprocessed = "0_preprocessed.dag";
static const std::string kEncCfg = "enc.cfg";
static const std::string kPts = "pt";
static const std::string kInUnenc = "ct_unenc";
static const std::string kInEnc = "ct_enc";
static const std::string kInEncClear = "ct_enc_clear";
static const std::string kOutEnc = "out_enc";
static const std::string kOutEncClear = "out_enc_clear";
static const std::string kOutUnenc = "out_unenc";
static const std::string kOutCheck = "out_tfhe";
static const std::string kOutUnencClear = "out_unenc_clear";

static const std::string kZeroCtName = "ZEROS";
static const std::string kMaskChunkIrKeyword = "MASK";
static const std::string kIndirectChunkIrKeyword = "INDIRECTION";

static const std::string kDslBootstrapC = "BootstrapC";
static const std::string kDslChetRepackC = "ChetRepackC";
static const std::string kDslRotateC = "HackRotateC";
static const std::string kDslMulCSI = "MulCSI";
static const std::string kDslAddCSI = "AddCSI";
static const std::string kDslAddCC = "AddCC";
static const std::string kDslAddCP = "AddCP";
static const std::string kDslMulCC = "MulCC";
static const std::string kDslMulCP = "MulCP";
static const std::string kDslInputC = "InputC";
static const std::string kDslOutputC = "OutputC";
static const std::string kDslReduceDimC = "ReduceDimC";
static const std::string kDslReorderDimsC = "ReorderDimC";
static const std::string kDslReplicateDimC = "ReplicateDimC";
static const std::string kDslDropDimC = "DropDimC";
static const std::string kDslInsertDimC = "InsertDimC";
static const std::string kDslResizeDimC = "ResizeDimC";
static const std::string kDslStrideDimC = "StrideDimC";
static const std::string kDslMergedStrideDimC = "MergedStrideDimC";
static const std::string kDslUnpaddedShiftC = "UnpaddedShiftC";
static const std::string kDslCyclicShiftC = "RotateC";

static const int kPrintDoublePrecision = 60;

static const std::string kCompiledProgram = "CompiledProgram";

static const LogChunkSize kDefaultLogChunkSize = 15;
static const int kDefaultLogScale = 50;
static const int kDefaultUsableLevels = 13;
static const BootstrappingPrecision kDefaultBootstrappingPrecision = 32;

}  // namespace fhelipe

#endif  // FHELIPE_CONSTANTS_H_
