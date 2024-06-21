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

#ifndef FHELIPE_DAG_IO_H_
#define FHELIPE_DAG_IO_H_

#include <cctype>
#include <memory>

#include "dag.h"
#include "include/extended_std.h"
#include "io_utils.h"
#include "t_op.h"

namespace fhelipe {

namespace details {

inline bool StreamEmpty(std::istream& stream) {
  while (std::isspace(stream.peek())) {
    stream.get();
  }
  stream.peek();
  return stream.eof();
}

template <class T>
std::vector<T> ExtractValuesAtIndices(const std::vector<T>& map,
                                      const std::vector<int>& indices) {
  std::vector<T> result;
  result.reserve(indices.size());
  for (int idx : indices) {
    result.push_back(map.at(idx));
  }
  return result;
}

template <class T>
std::vector<T> ReadParents(std::istream& ir_stream,
                           const std::unordered_map<int, T>& id_to_node) {
  auto parents = ReadStream<std::vector<int>>(ir_stream);
  return Estd::values_from_keys(id_to_node, parents);
}

template <class T>
Dag<T> ReadDag(
    std::istream& stream,
    const std::function<std::unique_ptr<T>(std::istream&)>& ReadDagNode) {
  Dag<T> dag;
  std::unique_ptr<T> new_node;
  std::unordered_map<int, std::shared_ptr<Node<T>>> id_to_node;
  while (!StreamEmpty(stream)) {
    int node_id = ReadStream<int>(stream);
    auto ancestors = ReadStream<std::vector<int>>(stream);
    auto new_node = ReadDagNode(stream);
    auto parent_nodes = ReadParents(stream, id_to_node);
    id_to_node.emplace(node_id, dag.AddNode(node_id, std::move(new_node),
                                            parent_nodes, ancestors));
  }
  return dag;
}

template <class T>
void WriteDag(std::ostream& stream, const Dag<T>& dag) {
  std::unordered_map<const Node<T>*, int> node_to_id;
  for (const auto& node : dag.NodesInTopologicalOrder()) {
    CHECK(!node->HoldsNothing());

    WriteStream<int>(stream, node->NodeId());
    stream << " ";

    WriteStream(stream, node->Ancestors());
    stream << " ";

    WriteStream<T>(stream, node->Value());
    stream << " ";

    std::vector<Node<T>*> parents =
        Estd::transform(node->Parents(), [](auto x) { return x.get(); });
    WriteStream(stream, Estd::values_from_keys(node_to_id, node->Parents()));
    stream << "\n";

    node_to_id.emplace(node.get(), node->NodeId());
  }
}

}  // namespace details

template <typename T>
struct IoStreamImpl<Dag<T>> {
  static void WriteStreamFunc(std::ostream& stream, const Dag<T>& dag) {
    details::WriteDag(stream, dag);
  }
  static Dag<T> ReadStreamFunc(std::istream& stream) {
    return details::ReadDag<T>(stream, [](std::istream& stream) {
      return std::make_unique<T>(ReadStream<T>(stream));
    });
  }
};

class TOpEmbrio;
class TOp;
class CtOp;

template <>
Dag<TOpEmbrio> ReadStream<Dag<TOpEmbrio>>(std::istream& stream);

template <>
Dag<TOp> ReadStream<Dag<TOp>>(std::istream& stream);

template <>
Dag<CtOp> ReadStream<Dag<CtOp>>(std::istream& stream);

template <>
void WriteStream<Dag<CtOp>>(std::ostream& stream, const Dag<CtOp>& dag);

}  // namespace fhelipe

#endif  // FHELIPE_DAG_READER_H_
