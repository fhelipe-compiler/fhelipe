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

#ifndef FHELIPE_EXE_FOLDER_GFLAG_UTILS_H_
#define FHELIPE_EXE_FOLDER_GFLAG_UTILS_H_

#include <gflags/gflags.h>

#include <filesystem>

#include "include/filesystem_utils.h"

DEFINE_string(exe_folder, "", "Path to executable directory");

namespace {

std::filesystem::path ExeFolderFromFlags() {
  CHECK(fhelipe::Exists(FLAGS_exe_folder))
      << FLAGS_exe_folder << " does not exist!";
  return {FLAGS_exe_folder};
}

}  // namespace
#endif  // FHELIPE_EXE_FOLDER_GFLAG_UTILS_H_
