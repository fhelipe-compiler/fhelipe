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

#ifndef FHELIPE_CLEARTEXT_H_
#define FHELIPE_CLEARTEXT_H_

#include <iosfwd>

#include "laid_out_chunk.h"
#include "level.h"
#include "level_info.h"
#include "leveled_t_op.h"
#include "plaintext.h"
#include "plaintext_chunk.h"
#include "program_context.h"
#include "scaled_pt_chunk.h"
#include "scaled_pt_val.h"

namespace fhelipe {

class Cleartext {
 public:
  Cleartext(const PtChunk& pt_chunk, const LevelInfo& level_info)
      : pt_chunk_(pt_chunk), level_info_(level_info) {}
  static Cleartext ZeroC(const ProgramContext& context,
                         const LevelInfo& level_info);

  Cleartext MulCC(const Cleartext& rhs) const;
  Cleartext MulCP(const ScaledPtChunk& chunk) const;
  Cleartext MulCS(const ScaledPtVal& scalar) const;

  Cleartext AddCC(const Cleartext& rhs) const;
  Cleartext AddCP(const ScaledPtChunk& chunk) const;
  Cleartext AddCS(const ScaledPtVal& scalar) const;

  Cleartext RotateC(int rotate_by) const;
  Cleartext RescaleC(LogScale rescale_amount) const;
  Cleartext BootstrapC(Level usable_levels) const;
  ChunkSize GetChunkSize() const { return pt_chunk_.size(); }

  PtChunk Decrypt() const { return pt_chunk_; }
  const LevelInfo& GetLevelInfo() const { return level_info_; }

 private:
  PtChunk pt_chunk_;
  LevelInfo level_info_;
};

template <>
Cleartext ReadStream<Cleartext>(std::istream& stream);

template <>
void WriteStream<Cleartext>(std::ostream& stream, const Cleartext& ct);

template <class T>
T Encrypt(const PtChunk& input_tensor, const ProgramContext& context);

template <class T>
T Encrypt(const PtChunk& input_tensor);

template <>
inline Cleartext Encrypt<Cleartext>(const PtChunk& input_tensor,
                                    const ProgramContext& context) {
  CHECK(input_tensor.size() == ChunkSize(context.GetLogChunkSize()));
  return {input_tensor, LevelInfo(context.UsableLevels(), context.LogScale())};
}

namespace detail {

template <>
inline void CheckLevelInfo<Cleartext>(
    const std::vector<LaidOutChunk<Cleartext>>& locs) {
  for (const auto& chunk : locs) {
    CHECK(chunk.Chunk().GetLevelInfo() == locs.at(0).Chunk().GetLevelInfo());
  }
}

}  // namespace detail

}  // namespace fhelipe

#endif  // FHELIPE_CLEARTEXT_H_
