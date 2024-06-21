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

#ifndef FHELIPE_DAG_H_
#define FHELIPE_DAG_H_

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "debug_info_archive.h"
#include "extended_std.h"
#include "node.h"
#include "zero_c.h"

namespace fhelipe {

template <class T>
class Dag {
 public:
  Dag() : sentinel_(std::make_shared<Node<T>>(std::unique_ptr<T>())) {}
  Dag(Dag&&) = default;
  Dag(const Dag&) = delete;
  Dag& operator=(const Dag&) = delete;
  Dag& operator=(Dag&&) = default;

  std::vector<std::shared_ptr<Node<T>>> NodesInTopologicalOrder() const;
  std::vector<std::shared_ptr<Node<T>>> NodesInReverseTopologicalOrder() const;
  std::vector<std::shared_ptr<Node<T>>> NodesInAncestorIdOrder() const;

  void AddInput(const std::shared_ptr<Node<T>>& input);

  // TODO(nsamar): These AddInput()'s should be convenience functions... the
  // shared_ptr AddInput() is the only one that is actually needed
  std::shared_ptr<Node<T>> AddInput(std::unique_ptr<T>&& input);
  std::shared_ptr<Node<T>> AddInput(std::unique_ptr<T>&& input,
                                    const std::vector<int>& ancestors);
  std::shared_ptr<Node<T>> AddInput(int node_id, std::unique_ptr<T>&& input);
  std::shared_ptr<Node<T>> AddInput(int node_id, std::unique_ptr<T>&& input,
                                    const std::vector<int>& ancestors);

  std::shared_ptr<Node<T>> GetNodeById(int node_id) const;

  // Old interface
  std::shared_ptr<Node<T>> AddNode(
      int node_id, std::unique_ptr<T>&& new_node,
      const std::vector<std::shared_ptr<Node<T>>>& parents,
      const std::vector<int>& ancestors);

  std::shared_ptr<Node<T>> AddNode(
      int node_id, std::unique_ptr<T>&& new_node,
      const std::vector<std::shared_ptr<Node<T>>>& parents);

  std::shared_ptr<Node<T>> AddNode(
      std::unique_ptr<T>&& new_node,
      const std::vector<std::shared_ptr<Node<T>>>& parents);

  std::shared_ptr<Node<T>> AddNode(
      std::unique_ptr<T>&& new_node,
      const std::vector<std::shared_ptr<Node<T>>>& parents,
      const std::vector<int>& ancestors);

  std::shared_ptr<Node<T>> Sentinel() { return sentinel_; }

 private:
  std::shared_ptr<Node<T>> sentinel_;

  mutable std::unordered_map<int, std::shared_ptr<Node<T>>> node_id_map_;

  void RebuildNodeIdMap() const;
};

template <class T>
void DagSanityCheck(const std::vector<std::shared_ptr<Node<T>>>& nodes);

template <class T>
class VisitedNodes {
 public:
  void Set(const Node<T>& node) { visited_.insert(&node); }
  bool IsVisited(const Node<T>& node) const { return visited_.contains(&node); }

 private:
  std::unordered_set<const Node<T>*> visited_;
};

template <class DagT, class VisitorT, typename FuncType>
void VisitInTopologicalOrder(
    const DagT& dag, VisitorT& visitor,
    const FuncType& hook = [](const auto& x, int linum) {}) {
  int linum = 0;
  for (const auto& node : dag.NodesInTopologicalOrder()) {
    if (node) {
      node->Accept(visitor);
      hook(*node, ++linum);
    }
  }
}

template <class NodeT, class T>
std::vector<std::shared_ptr<Node<NodeT>>> CollectNodesByTypeInTopologicalOrder(
    const Dag<NodeT>& dag) {
  std::vector<std::shared_ptr<Node<NodeT>>> result;
  for (auto& node : dag.NodesInTopologicalOrder()) {
    if (dynamic_cast<const T*>(&node->Value().GetTOp())) {
      result.push_back(node);
    }
  }
  return result;
}

template <class T>
void Dag<T>::RebuildNodeIdMap() const {
  for (const auto& node : NodesInTopologicalOrder()) {
    node_id_map_.emplace(node->NodeId(), node);
  }
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::GetNodeById(int node_id) const {
  if (Estd::contains_key(node_id_map_, node_id)) {
    return node_id_map_.at(node_id);
  }
  RebuildNodeIdMap();
  return node_id_map_.at(node_id);
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddInput(std::unique_ptr<T>&& input,
                                          const std::vector<int>& ancestors) {
  std::shared_ptr<Node<T>> result =
      std::make_shared<Node<T>>(std::move(input), ancestors);
  AddParentChildEdge(sentinel_, result);
  return result;
}

template <class T>
void Dag<T>::AddInput(const std::shared_ptr<Node<T>>& input) {
  AddParentChildEdge(sentinel_, input);
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddNode(
    std::unique_ptr<T>&& new_node,
    const std::vector<std::shared_ptr<Node<T>>>& parents,
    const std::vector<int>& ancestors) {
  if (parents.empty()) {
    return AddInput(std::move(new_node), ancestors);
  }
  return CreateChild(std::move(new_node), parents, ancestors);
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddNode(
    std::unique_ptr<T>&& new_node,
    const std::vector<std::shared_ptr<Node<T>>>& parents) {
  return AddNode(std::move(new_node), parents, {});
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddNode(
    int node_id, std::unique_ptr<T>&& new_node,
    const std::vector<std::shared_ptr<Node<T>>>& parents,
    const std::vector<int>& ancestors) {
  if (parents.empty()) {
    return AddInput(node_id, std::move(new_node), ancestors);
  }
  return CreateChild(node_id, std::move(new_node), parents, ancestors);
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddNode(
    int node_id, std::unique_ptr<T>&& new_node,
    const std::vector<std::shared_ptr<Node<T>>>& parents) {
  return AddNode(node_id, std::move(new_node), parents, {});
}

template <class T>
bool AllParentsVisited(const VisitedNodes<T>& visited, const Node<T>& node) {
  return Estd::all_of(node.Parents(), [&visited](const auto& parent) {
    return visited.IsVisited(*parent);
  });
}

template <class T>
std::vector<std::shared_ptr<Node<T>>> Dag<T>::NodesInReverseTopologicalOrder()
    const {
  return Estd::reverse(NodesInTopologicalOrder());
}

template <class T>
struct ByAncestorId {
  bool operator()(const std::shared_ptr<Node<T>>& a,
                  const std::shared_ptr<Node<T>>& b) const {
    auto a_aid = a->Ancestors().empty() ? 0 : a->Ancestors()[0];
    auto b_aid = b->Ancestors().empty() ? 0 : b->Ancestors()[0];
    return a_aid < b_aid;
  }
};

template <class T>
std::vector<std::shared_ptr<Node<T>>> Dag<T>::NodesInAncestorIdOrder() const {
  auto nodes = NodesInTopologicalOrder();
  std::sort(nodes.begin(), nodes.end(), ByAncestorId<T>());
  return nodes;
}

template <class T>
std::vector<std::shared_ptr<Node<T>>> Dag<T>::NodesInTopologicalOrder() const {
  std::vector<std::shared_ptr<Node<T>>> nodes;
  std::queue<std::shared_ptr<Node<T>>> frontier;
  frontier.push(sentinel_);

  VisitedNodes<T> visited;
  visited.Set(*sentinel_);

  // Topo sort
  while (!frontier.empty()) {
    auto vertex = frontier.front();
    frontier.pop();
    if (vertex != sentinel_) {
      nodes.push_back(vertex);
    }

    for (auto child : vertex->Children()) {
      if (AllParentsVisited(visited, *child) && !visited.IsVisited(*child)) {
        frontier.push(child);
        visited.Set(*child);
      }
    }
  }

  return nodes;
}

template <class T>
void DagSanityCheck(const std::vector<std::shared_ptr<Node<T>>>& nodes) {
  // Check parent->child relationships are consistent
  for (const auto& node : nodes) {
    for (const auto& child : node->Children()) {
      CHECK(Estd::contains(child->Parents(), node));
    }
    for (const auto& parent : node->Parents()) {
      CHECK(Estd::contains(parent->Children(), node));
    }
  }

  // Check DAG
  VisitedNodes<T> visited;
  for (const auto& node : nodes) {
    for (const auto& parent : node->Parents()) {
      CHECK(visited.IsVisited(*parent));
    }
    for (const auto& child : node->Children()) {
      CHECK(!visited.IsVisited(*child));
    }
    visited.Set(*node);
  }
}

template <class T>
DebugInfoArchive DagToDebugInfoArchive(const Dag<T>& dag) {
  std::unordered_map<int, std::vector<int>> result;
  for (const auto& node : dag.NodesInTopologicalOrder()) {
    result.emplace(node->NodeId(), node->Ancestors());
  }
  return DebugInfoArchive{result};
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddInput(int node_id,
                                          std::unique_ptr<T>&& input,
                                          const std::vector<int>& ancestors) {
  std::shared_ptr<Node<T>> result =
      std::make_shared<Node<T>>(node_id, std::move(input), ancestors);
  AddParentChildEdge(sentinel_, result);
  return result;
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddInput(int node_id,
                                          std::unique_ptr<T>&& input) {
  return AddInput(node_id, input, {});
}

template <class T>
std::shared_ptr<Node<T>> Dag<T>::AddInput(std::unique_ptr<T>&& input) {
  return AddInput(std::move(input), {});
}

template <class T>
Dag<T> CloneFromAncestor(const Dag<T>& in_dag) {
  std::unordered_map<const Node<T>*, std::shared_ptr<Node<T>>> old_to_new_nodes;
  Dag<T> out_dag;
  for (const auto& old_node : in_dag.NodesInTopologicalOrder()) {
    const auto parents = ExtractParents(old_to_new_nodes, *old_node);
    auto new_node = out_dag.AddNode(old_node->Value().CloneUniq(), parents,
                                    {old_node->NodeId()});
    old_to_new_nodes.emplace(old_node.get(), new_node);
  }
  return out_dag;
}

}  // namespace fhelipe

#endif  //  FHELIPE_DAG_H_
