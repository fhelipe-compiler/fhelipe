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

#ifndef FHELIPE_DEBUG_INFO_H_
#define FHELIPE_DEBUG_INFO_H_

#include <list>
#include <memory>
#include <unordered_map>

#include "dag.h"
#include "extended_std.h"
#include "io_utils.h"

namespace fhelipe {

using SrcToDestSetBijection =
    std::vector<std::pair<std::set<int>, std::set<int>>>;

template <typename SrcDagT, typename DestDagT, typename SrcT, typename DestT>
class DebugInfo {
 public:
  DebugInfo(const SrcDagT* src_dag, const DestDagT* dest_dag,
            const SetBijection& bijection)
      : src_dag_(src_dag), dest_dag_(dest_dag), bijection_(bijection) {
    for (const auto& [src, dest] : bijection.Mappings()) {
      std::vector<const Node<SrcT>*> src_vec =
          Estd::transform(Estd::set_to_vector(src.Values()),
                          [this](int node_id) -> const Node<SrcT>* {
                            return src_dag_->GetNodeById(node_id).get();
                          });
      std::vector<const Node<DestT>*> dest_vec =
          Estd::transform(Estd::set_to_vector(dest.Values()),
                          [this](int node_id) -> const Node<DestT>* {
                            return dest_dag_->GetNodeById(node_id).get();
                          });
      mappings_.emplace_back(src_vec, dest_vec);
    }
  }

  const SetBijection& Bijection() const { return bijection_; }
  const SrcDagT* SourceDag() const { return src_dag_; }
  const DestDagT* DestinationDag() const { return dest_dag_; }
  const std::vector<std::pair<std::vector<const Node<SrcT>*>,
                              std::vector<const Node<DestT>*>>>&
  Mappings() const {
    return mappings_;
  }

 private:
  const SrcDagT* src_dag_;
  const DestDagT* dest_dag_;
  SetBijection bijection_;
  std::vector<std::pair<std::vector<const Node<SrcT>*>,
                        std::vector<const Node<DestT>*>>>
      mappings_;
};

}  // namespace fhelipe

#endif  // FHELIPE_DEBUG_INFO_H_
