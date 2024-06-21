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

#ifndef FHELIPE_SCALED_PT_CHUNK_H_
#define FHELIPE_SCALED_PT_CHUNK_H_

#include "include/chunk.h"
#include "include/log_scale.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"

namespace fhelipe {

class ScaledPtChunk {
 public:
  ScaledPtChunk(LogScale log_scale, const PtChunk& chunk)
      : log_scale_(log_scale), chunk_(chunk) {}

  LogScale GetLogScale() const { return log_scale_; }
  const PtChunk& chunk() const { return chunk_; }

 private:
  LogScale log_scale_;
  PtChunk chunk_;
};

}  // namespace fhelipe

#endif  // FHELIPE_SCALED_PT_CHUNK_H_
