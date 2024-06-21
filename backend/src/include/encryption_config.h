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

#ifndef FHELIPE_ENCRYPTION_CONFIG_H_
#define FHELIPE_ENCRYPTION_CONFIG_H_

#include <fstream>
#include <string>
#include <vector>

#include "dag.h"
#include "dictionary.h"
#include "t_op.h"
#include "tensor_layout.h"

namespace fhelipe {
class TOp;
template <class T>
class Dag;

class EncryptionConfig {
 public:
  EncryptionConfig(bool is_input, const std::string& name,
                   const TensorLayout& layout)
      : is_input_(is_input), name_(name), layout_(layout) {}
  const TensorLayout& Layout() const { return layout_; }
  bool IsInput() const { return is_input_; }
  const std::string& Name() const { return name_; }

 private:
  bool is_input_;
  std::string name_;
  TensorLayout layout_;
};

template <>
EncryptionConfig ReadStream<EncryptionConfig>(std::istream& cfg_stream);

template <>
void WriteStream<EncryptionConfig>(std::ostream& stream,
                                   const EncryptionConfig& config);

void AddEncryptionConfigs(Dictionary<EncryptionConfig>& config_dict,
                          const Dag<TOp>& top_dag);

}  // namespace fhelipe

#endif  // FHELIPE_ENCRYPTION_CONFIG_H_
