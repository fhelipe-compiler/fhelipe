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

#include "include/filesystem_utils.h"

#include <glog/logging.h>

#include <filesystem>
#include <system_error>

namespace fhelipe {

void EnsureDirectoryExists(const std::filesystem::path& path) {
  if (Exists(path)) {
    return;
  }
  std::error_code error_code;
  CHECK(std::filesystem::create_directories(path, error_code))
      << "Failed to create directory " << path
      << ". Returned error code: " << error_code;
  CHECK(!error_code);
}

bool Exists(const std::filesystem::path& path) {
  std::error_code error_code;
  bool result = std::filesystem::exists(path, error_code);
  CHECK(!error_code);
  return result;
}

std::vector<std::filesystem::path> ContainedFilepaths(
    const std::filesystem::path& folder_path) {
  std::vector<std::filesystem::path> result;
  for (const auto& dir_entry :
       std::filesystem::recursive_directory_iterator(folder_path)) {
    result.push_back(dir_entry.path());
  }
  return result;
}

void EnsureDoesNotExist(const std::filesystem::path& path) {
  if (!Exists(path)) {
    return;
  }
  std::error_code error_code;
  std::filesystem::remove_all(path, error_code);
  CHECK(!error_code) << path;
}

std::string ReadFileContent(const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

}  // namespace fhelipe
