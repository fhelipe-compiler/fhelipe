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

#include "include/basic_preprocessor.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#include "include/constants.h"
#include "include/filesystem_utils.h"
#include "include/io_utils.h"
#include "include/level.h"
#include "include/log_scale.h"
#include "include/pass_utils.h"

namespace fhelipe {

namespace {

const std::string kLogScaleFlag = "~";
const std::string kMaxUsableLevelsFlag = "#";

std::string SedCommand(std::string content, const std::string& from,
                       const std::string& to) {
  std::regex from_regex(from);
  return std::regex_replace(content, from_regex, to);
}

}  // namespace

PreprocessorOutput BasicPreprocessor::DoPass(const PreprocessorInput& code) {
  std::string result = SedCommand(code, kLogScaleFlag,
                                  std::to_string(default_log_scale_.value()));
  result = SedCommand(result, kMaxUsableLevelsFlag,
                      std::to_string(max_usable_level_.value()));
  int linum = -1;
  std::string preprocessed;
  for (size_t curr = result.find('\n'); curr != std::string::npos;
       curr = result.find('\n')) {
    preprocessed += (std::to_string(++linum) + " 0 ");
    preprocessed += result.substr(0, curr + 1);
    result = result.substr(curr + 1, result.size() - curr - 1);
  }
  preprocessed += result;

  if (linum < 1) {
    LOG(FATAL) << "Empty frontend input!";
  }

  return preprocessed;
}

}  // namespace fhelipe
