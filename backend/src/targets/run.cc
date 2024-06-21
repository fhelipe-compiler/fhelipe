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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/cleartext.h"
#include "include/constants.h"
#include "include/ct_program.h"
#include "include/evaluator.h"
#include "include/glue_code.h"
#include "include/io_manager.h"
#include "include/lattigo_ct.h"
#include "include/persisted_dictionary.h"
#include "include/plaintext.h"
#include "targets/gflag_utils/ciphertext_type_gflag_utils.h"
#include "targets/gflag_utils/exe_folder_gflag_utils.h"

using namespace fhelipe;

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::ios_base::sync_with_stdio(false);

  auto exe_folder = ExeFolderFromFlags();
  CiphertextType ct_type = CiphertextTypeFromFlags();

  if (ct_type == CiphertextType::Cleartext) {
    Run<Cleartext>(exe_folder);
  } else if (ct_type == CiphertextType::LattigoCt) {
    Run<LattigoCt>(exe_folder);
  } else {
    LOG(FATAL) << "ct_type not recognized!";
  }
}
