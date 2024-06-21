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

#ifndef FHELIPE_CIPHERTEXT_TYPE_GFLAG_UTILS_H_
#define FHELIPE_CIPHERTEXT_TYPE_GFLAG_UTILS_H_

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(
    ct_type, "clear",
    "Type of ciphertext to be used during program execution or encryption.");

namespace {

enum class CiphertextType { Cleartext, LattigoCt};
CiphertextType CiphertextTypeFromFlags();

CiphertextType CiphertextTypeFromFlags() {
  if (FLAGS_ct_type == "clear") {
    return CiphertextType::Cleartext;
  }
  if (FLAGS_ct_type == "lattigo") {
    return CiphertextType::LattigoCt;
  }
  LOG(FATAL) << "Unrecognized ciphertext type " << FLAGS_ct_type;
}

}  // namespace
#endif  // FHELIPE_CIPHERTEXT_TYPE_GFLAG_UTILS_H_
