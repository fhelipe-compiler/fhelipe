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

#include "include/cleartext.h"

#include <vector>

#include "include/checker.h"
#include "include/chunk.h"
#include "include/constants.h"
#include "include/io_utils.h"
#include "include/level_info_utils.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"
#include "include/program_context.h"
#include "include/scaled_pt_chunk.h"
#include "include/scaled_pt_val.h"

namespace fhelipe {

namespace {}  // namespace

Cleartext Cleartext::BootstrapC(Level usable_levels) const {
  CheckSmallEnoughForBootstrapping(Decrypt().Values());
  return {this->pt_chunk_,
          LevelInfo{usable_levels, this->level_info_.LogScale()}};
}

Cleartext Cleartext::ZeroC(const ProgramContext& context,
                           const LevelInfo& level_info) {
  auto pt_chunk =
      PtChunk(std::vector<PtVal>(1 << context.GetLogChunkSize().value()));
  return Cleartext{pt_chunk, level_info};
}

Cleartext Cleartext::RescaleC(LogScale rescale_amount) const {
  return {this->pt_chunk_,
          LevelInfo{this->GetLevelInfo().Level() - Level(1),
                    this->GetLevelInfo().LogScale() - rescale_amount}};
}

Cleartext Cleartext::MulCC(const Cleartext& rhs) const {
  return {
      Mul(pt_chunk_, rhs.pt_chunk_),
      LevelInfo(
          std::min(this->GetLevelInfo().Level(), rhs.GetLevelInfo().Level()),
          this->GetLevelInfo().LogScale() + rhs.GetLevelInfo().LogScale())};
}

Cleartext Cleartext::MulCP(const ScaledPtChunk& pt_chunk) const {
  return {Mul(pt_chunk_, pt_chunk.chunk()),
          LevelInfo{this->GetLevelInfo().Level(),
                    this->GetLevelInfo().LogScale() + pt_chunk.GetLogScale()}};
}

Cleartext Cleartext::MulCS(const ScaledPtVal& scalar) const {
  return {MulScalar(pt_chunk_, scalar.value()),
          LevelInfo{this->GetLevelInfo().Level(),
                    this->GetLevelInfo().LogScale() + scalar.GetLogScale()}};
}

Cleartext Cleartext::AddCC(const Cleartext& rhs) const {
  return {Add(pt_chunk_, rhs.pt_chunk_),
          LevelInfo{std::min(this->GetLevelInfo().Level(),
                             rhs.GetLevelInfo().Level()),
                    std::max(this->GetLevelInfo().LogScale(),
                             rhs.GetLevelInfo().LogScale())}};
}

Cleartext Cleartext::AddCP(const ScaledPtChunk& pt_chunk) const {
  return {Add(pt_chunk_, pt_chunk.chunk()),
          LevelInfo{this->GetLevelInfo().Level(),
                    LogScale(std::max(this->GetLevelInfo().LogScale().value(),
                                      pt_chunk.GetLogScale().value()))}};
}

Cleartext Cleartext::AddCS(const ScaledPtVal& scalar) const {
  return {AddScalar(pt_chunk_, scalar.value()),
          LevelInfo{this->GetLevelInfo().Level(),
                    this->GetLevelInfo().LogScale() + scalar.GetLogScale()}};
}

Cleartext Cleartext::RotateC(int rotate_by) const {
  return {Rotate(pt_chunk_, rotate_by), this->GetLevelInfo()};
}

template <>
Cleartext ReadStream<Cleartext>(std::istream& stream) {
  auto level_info = ReadStream<LevelInfo>(stream);
  return {PtChunk(ReadStream<std::vector<PtVal>>(stream)), level_info};
}

template <>
void WriteStream<Cleartext>(std::ostream& stream, const Cleartext& ct) {
  WriteStream(stream, ct.GetLevelInfo());
  WriteStream(stream, ct.Decrypt().Values());
}

}  // namespace fhelipe
