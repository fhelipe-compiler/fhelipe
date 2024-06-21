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

#ifndef FHELIPE_FILESYSTEM_UTILS_H_
#define FHELIPE_FILESYSTEM_UTILS_H_

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#include "io_utils.h"

namespace fhelipe {

void EnsureDirectoryExists(const std::filesystem::path& path);
bool Exists(const std::filesystem::path& path);

void EnsureDoesNotExist(const std::filesystem::path& path);

template <typename T>
void WriteFile(const std::filesystem::path& path, const T& value) {
  auto stream = OpenStream<std::ofstream>(path);
  WriteStream<T>(stream, value);
}

template <typename T>
T ReadFile(const std::filesystem::path& path) {
  auto stream = OpenStream<std::ifstream>(path);
  return ReadStream<T>(stream);
}

std::string ReadFileContent(const std::filesystem::path& path);

std::vector<std::filesystem::path> ContainedFilepaths(
    const std::filesystem::path& folder_path);

template <>
inline void WriteStream<std::filesystem::path>(
    std::ostream& stream, const std::filesystem::path& path) {
  WriteStream<std::string>(stream, path.string());
}

template <>
inline std::filesystem::path ReadStream<std::filesystem::path>(
    std::istream& stream) {
  return {ReadStream<std::string>(stream)};
}

}  // namespace fhelipe

#endif  // FHELIPE_FILESYSTEM_UTILS_H_
