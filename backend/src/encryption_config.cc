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

#include "include/encryption_config.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "include/constants.h"
#include "include/dag.h"
#include "include/dictionary.h"
#include "include/dimension_bit.h"
#include "include/io_utils.h"
#include "include/t_input_c.h"
#include "include/t_op.h"
#include "include/t_output_c.h"
#include "include/tensor_layout.h"
#include "include/utils.h"

namespace fhelipe {

std::optional<DimensionBit> ReadOptDimBit(std::ifstream& is) {
  auto token = ReadStream<std::string>(is);
  if (token == kInvalidOptionalToken) {
    return std::nullopt;
  }
  auto iss = std::istringstream(token);
  auto dimension = ReadStream<int>(iss);
  auto bit_index = ReadStream<int>(iss);
  return std::make_optional(DimensionBit(dimension, bit_index));
}

std::vector<std::optional<DimensionBit>> ReadLayoutBits(
    std::ifstream& cfg_stream) {
  std::vector<std::optional<DimensionBit>> result;
  auto sz = ReadStream<int>(cfg_stream);
  result.reserve(sz);
  while (result.size() < sz) {
    std::optional<DimensionBit> value = ReadOptDimBit(cfg_stream);
    result.push_back(value);
  }
  return result;
}

template <>
EncryptionConfig ReadStream<EncryptionConfig>(std::istream& cfg_stream) {
  std::string tensor_name;
  bool is_input;
  auto token = ReadStream<std::string>(cfg_stream);
  if (token == "Input") {
    is_input = true;
  } else if (token == "Output") {
    is_input = false;
  } else {
    LOG(FATAL) << "Invalid encryption config!";
  }
  auto name = ReadStream<std::string>(cfg_stream);
  const auto& layout = ReadStream<TensorLayout>(cfg_stream);
  return {is_input, name, layout};
}

bool ClearStreamWhiteSpace(std::ifstream& io_config) {
  while (io_config.peek() != EOF && std::isspace(io_config.peek())) {
    io_config.get();
  }
  return io_config.peek() != EOF;
}

void AddEncryptionConfigs(Dictionary<EncryptionConfig>& config_dict,
                          const Dag<TOp>& top_dag) {
  for (const auto& node_t : top_dag.NodesInTopologicalOrder()) {
    const auto* node = &node_t->Value();
    if (const auto* input_c = dynamic_cast<const TInputC*>(node)) {
      config_dict.Record(
          input_c->Name(),
          EncryptionConfig(true, input_c->Name(), input_c->OutputLayout()));
    }
    if (const auto* output_c = dynamic_cast<const TOutputC*>(node)) {
      config_dict.Record(
          output_c->Name(),
          EncryptionConfig(false, output_c->Name(), output_c->OutputLayout()));
    }
  }
}

template <>
void WriteStream<EncryptionConfig>(std::ostream& stream,
                                   const EncryptionConfig& cfg) {
  if (cfg.IsInput()) {
    stream << "Input ";
  } else {
    stream << "Output ";
  }
  stream << cfg.Name() << " ";
  WriteStream(stream, cfg.Layout());
}

}  // namespace fhelipe
