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

#ifndef FHELIPE_DICTIONARY_IMPL_H_
#define FHELIPE_DICTIONARY_IMPL_H_

// nsamar: I need this dicitonary_impl.h file separate from dictionary.h
// becasue CreateInstance needs to know that PersistedDictionary and
// RamDictionary are derived from Dictionary... to know this, CreateInstance
// must be aware of persisted_dictionary.h and ram_dictionary.h. but
// persisted_dictionary.h and ram_dictionary.h need to be aware of dictionary.h.
// So the dicitonary_impl.h breaks the cyclic dependence.

#include "dictionary.h"
#include "persisted_dictionary.h"
#include "ram_dictionary.h"

namespace fhelipe {

template <class T>
std::unique_ptr<Dictionary<T>> Dictionary<T>::CreateInstance(
    std::istream& stream) {
  std::string token = ReadStream<std::string>(stream);
  if (token == PersistedDictionary<T>::StaticTypeName()) {
    return ReadStreamWithoutTypeNamePrefix<PersistedDictionary<T>, T>(stream)
        .CloneUniq();
  }
  if (token == RamDictionary<T>::StaticTypeName()) {
    return ReadStreamWithoutTypeNamePrefix<RamDictionary<T>, T>(stream)
        .CloneUniq();
  }
  LOG(FATAL) << "Unrecognized token " << token;
}

}  // namespace fhelipe

#endif  // FHELIPE_DICTIONARY_IMPL_H_
