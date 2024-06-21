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

#include <glog/logging.h>

#include <algorithm>
#include <ostream>
#include <string>
#include <vector>

#include "include/cleartext.h"
#include "include/constants.h"
#include "include/glue_code.h"
#include "include/lattigo_ct.h"
#include "include/packer.h"
#include "include/plaintext.h"
#include "latticpp/ckks/lattigo_param.h"
#include "targets/gflag_utils/ciphertext_type_gflag_utils.h"
#include "targets/gflag_utils/exe_folder_gflag_utils.h"

using namespace fhelipe;
using namespace latticpp;

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::ios_base::sync_with_stdio(false);

  auto exe_folder = ExeFolderFromFlags();
  CiphertextType ct_type = CiphertextTypeFromFlags();

  std::cout << "exe_folder: " << exe_folder << std::endl;
  if (ct_type == CiphertextType::Cleartext) {
    Decrypt<Cleartext>(exe_folder);
  } else if (ct_type == CiphertextType::LattigoCt) {
    Decrypt<LattigoCt>(exe_folder);
  } else {
    LOG(FATAL) << "ct_type not recognized!";
  }
}
