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

#ifndef FHELIPE_PLAINTEXT_CHUNK_H_
#define FHELIPE_PLAINTEXT_CHUNK_H_

#include "chunk.h"
#include "io_utils.h"
#include "plaintext.h"

namespace fhelipe {

using PtChunk = Chunk<PtVal>;

template <>
inline PtChunk ReadStream<PtChunk>(std::istream& stream) {
  return PtChunk{ReadStream<std::vector<PtVal>>(stream)};
}

template <>
inline void WriteStream<PtChunk>(std::ostream& stream, const PtChunk& chunk) {
  WriteStream(stream, chunk.Values());
}

}  // namespace fhelipe

#endif  // FHELIPE_PLAINTEXT_CHUNK_H_
