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

#ifndef FHELIPE_NODE_H_
#define FHELIPE_NODE_H_

#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include "include/extended_std.h"

namespace fhelipe {

template <class T>
class Node {
 public:
  Node() = default;
  Node(Node&&) noexcept = default;
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  Node& operator=(Node&&) noexcept = default;

  ~Node() = default;

  explicit Node(std::unique_ptr<T>&& value);
  Node(int node_id, std::unique_ptr<T>&& value);
  Node(int node_id, std::unique_ptr<T>&& value,
       const std::vector<int>& ancestors);
  Node(std::unique_ptr<T>&& value, const std::vector<int>& ancestors);

  void AddParent(const std::shared_ptr<Node<T>>& parent);
  void AddChild(const std::shared_ptr<Node<T>>& child);

  void RemoveChild(const Node<T>& child);
  void RemoveParent(const Node<T>& parent);
  bool ContainsChild(const Node<T>& candidate);

  const T& Value() const;
  T& Value();
  void SetValue(std::unique_ptr<T>&& new_value);

  int NodeId() const;
  void RemoveAllParents() { parents_ = {}; }

  bool HoldsNothing() const { return !bool(value_); }
  bool IsSentinel() const { return HoldsNothing(); }

  std::vector<std::shared_ptr<Node<T>>> Parents() const;
  const std::set<std::shared_ptr<Node<T>>>& Children() const;

  std::shared_ptr<Node<T>> Sentinel() const;

  const std::vector<int>& Ancestors() const { return ancestor_node_ids_; }

  void AddAncestor(int ancestor_id) {
    ancestor_node_ids_.push_back(ancestor_id);
  }

  bool IsParent(const Node<T>& candidate) const;
  bool IsDoubleParent(const Node<T>& candidate) const;

 private:
  int node_id_;
  std::unique_ptr<T> value_;
  std::vector<std::weak_ptr<Node<T>>> parents_;
  std::set<std::shared_ptr<Node<T>>> children_;
  std::vector<int> ancestor_node_ids_;

  static int max_used_node_ids_;
};

template <typename T>
void ReplaceParent(std::shared_ptr<Node<T>> me, Node<T>& replace_me,
                   const std::shared_ptr<Node<T>>& new_parent);

template <class T>
std::shared_ptr<Node<T>> CloneUniqWithSameAncestors(const Node<T>& node) {
  return std::make_shared<Node<T>>(node.Value().CloneUniq(), node.Ancestors());
}

template <class T>
void AddNodeOnParentChildEdge(std::shared_ptr<Node<T>> parent,
                              std::shared_ptr<Node<T>> child,
                              std::shared_ptr<Node<T>> new_node) {
  AddParentChildEdge(parent, new_node);
  AddParentChildEdge(new_node, child);
  RemoveParentChildEdge(*parent, *child);
}

template <class T>
void InheritChildren(Node<T>& parent, std::shared_ptr<Node<T>> node) {
  auto children = Estd::set_to_vector(parent.Children());
  for (auto child : children) {
    ReplaceParent(child, parent, node);
  }
}

template <class T>
bool IsParentChildEdge(const std::shared_ptr<Node<T>>& parent,
                       const std::shared_ptr<Node<T>>& child) {
  return Estd::contains(parent->Children(), child) &&
         Estd::contains(child->Parents(), parent);
}

template <class T>
void SwapParentAndChild(std::shared_ptr<Node<T>> parent,
                        std::shared_ptr<Node<T>> child) {
  CHECK(parent->Children().size() == 1);
  CHECK(Estd::vector_to_set(parent->Parents()).size() == 1);
  CHECK(Estd::vector_to_set(child->Parents()).size() == 1);
  CHECK(IsParentChildEdge(parent, child));

  RemoveNode(*parent);

  InheritChildren(*child, parent);
  AddParentChildEdge(child, parent);
}

template <class T>
int Node<T>::max_used_node_ids_ = 0;

template <class T>
void AddParentChildEdge(const std::shared_ptr<Node<T>>& parent,
                        const std::shared_ptr<Node<T>>& child);

template <class T>
void RemoveParentChildEdge(Node<T>& parent, Node<T>& child);

template <class T>
bool Node<T>::ContainsChild(const Node<T>& candidate) {
  for (const auto& child : children_) {
    if (&candidate == child.get()) {
      return true;
    }
  }
  return false;
}

template <class T>
void ReplaceParent(std::shared_ptr<Node<T>> me, Node<T>& replace_me,
                   const std::shared_ptr<Node<T>>& new_parent) {
  CHECK(new_parent.get());
  CHECK(me.get());
  CHECK(&replace_me != new_parent.get());
  AddParentChildEdge(new_parent, me);
  if (me->IsDoubleParent(replace_me)) {
    me->AddParent(new_parent);
  }

  RemoveParentChildEdge(replace_me, *me);
}

template <class T>
bool Node<T>::IsDoubleParent(const Node<T>& candidate) const {
  std::vector<const Node<T>*> tmp_parents = Estd::transform(
      parents_,
      [](const auto& ptr) -> const Node<T>* { return ptr.lock().get(); });
  return Estd::count(tmp_parents, &candidate) == 2;
}

template <class T>
bool Node<T>::IsParent(const Node<T>& candidate) const {
  std::vector<const Node<T>*> tmp_parents = Estd::transform(
      parents_,
      [](const auto& ptr) -> const Node<T>* { return ptr.lock().get(); });
  return Estd::contains(tmp_parents, &candidate);
}

template <class T>
std::shared_ptr<Node<T>> Node<T>::Sentinel() const {
  CHECK(!parents_.empty());
  // Assumes sentinel holds nothing
  auto sentinel_candidate = parents_.at(0).lock();
  if (sentinel_candidate->IsSentinel()) {
    return sentinel_candidate;
  }
  return sentinel_candidate->Sentinel();
}

template <class T>
std::shared_ptr<Node<T>> CreateChild(
    std::unique_ptr<T>&& new_value,
    const std::vector<std::shared_ptr<Node<T>>>& parents);

template <class T>
std::shared_ptr<Node<T>> CreateChild(
    int node_id, std::unique_ptr<T>&& new_value,
    const std::vector<std::shared_ptr<Node<T>>>& parents);

template <class T>
std::shared_ptr<Node<T>> CreateChild(
    int node_id, std::unique_ptr<T>&& new_value,
    const std::vector<std::shared_ptr<Node<T>>>& parents,
    const std::vector<int>& ancestors);

template <class T>
void RemoveNode(Node<T>& node);

template <class T>
void Node<T>::RemoveChild(const Node<T>& node) {
  auto children = Estd::set_to_vector(children_);
  for (const auto& child : children) {
    if (child.get() == &node) {
      children_.erase(child);
      return;
    }
  }
  LOG(FATAL);
}

template <class T>
void Node<T>::RemoveParent(const Node<T>& node) {
  std::vector<const Node<T>*> tmp_parents = Estd::transform(
      parents_,
      [](const auto& ptr) -> const Node<T>* { return ptr.lock().get(); });
  CHECK(Estd::contains(tmp_parents, &node));
  int value_idx = Estd::find_index(tmp_parents, &node);

  for (int idx = value_idx; idx < parents_.size() - 1; ++idx) {
    parents_.at(idx) = parents_.at(idx + 1);
  }
  parents_.pop_back();
  if (IsParent(node)) {
    // Remove node again in case of aliasing
    RemoveParent(node);
  }
}

template <class T>
void RemoveNodeWithoutReassaigningChildren(
    const std::shared_ptr<Node<T>>& node) {
  auto children = Estd::set_to_vector(node->Children());
  for (auto child : children) {
    RemoveParentChildEdge(*node, *child);
  }
  for (auto parent :
       Estd::set_to_vector(Estd::vector_to_set(node->Parents()))) {
    RemoveParentChildEdge(*parent, *node);
  }
}

template <class T>
void RemoveNode(Node<T>& node) {
  // Don't know how to reassign orphaned children otherwise
  CHECK(node.Parents().size() == 1);

  const auto parent = node.Parents().at(0);

  InheritChildren(node, parent);

  RemoveParentChildEdge(*parent, node);
}

template <class T>
std::shared_ptr<Node<T>> CreateChild(
    std::unique_ptr<T>&& new_value,
    const std::vector<std::shared_ptr<Node<T>>>& parents,
    const std::vector<int>& ancestors) {
  CHECK(!parents.empty());
  auto new_node = std::make_shared<Node<T>>(std::move(new_value), ancestors);
  for (auto parent : parents) {
    AddParentChildEdge(parent, new_node);
  }
  return new_node;
}

template <class T>
std::shared_ptr<Node<T>> CreateChild(
    std::unique_ptr<T>&& new_value,
    const std::vector<std::shared_ptr<Node<T>>>& parents) {
  return CreateChild(std::move(new_value), parents, {});
}

template <class T>
std::shared_ptr<Node<T>> CreateChild(
    int node_id, std::unique_ptr<T>&& new_value,
    const std::vector<std::shared_ptr<Node<T>>>& parents,
    const std::vector<int>& ancestors) {
  CHECK(!parents.empty());
  auto new_node =
      std::make_shared<Node<T>>(node_id, std::move(new_value), ancestors);
  for (auto parent : parents) {
    AddParentChildEdge(parent, new_node);
  }
  return new_node;
}

template <class T>
std::shared_ptr<Node<T>> CreateChild(
    int node_id, std::unique_ptr<T>&& new_value,
    const std::vector<std::shared_ptr<Node<T>>>& parents) {
  return CreateChild(node_id, std::move(new_value), parents, {});
}

template <class T>
void Node<T>::AddParent(const std::shared_ptr<Node<T>>& parent) {
  parents_.push_back(parent);
}

template <class T>
void Node<T>::AddChild(const std::shared_ptr<Node<T>>& child) {
  children_.insert(child);
}

template <class T>
const T& Node<T>::Value() const {
  return *value_;
}

template <class T>
T& Node<T>::Value() {
  return *value_;
}

template <class T>
void Node<T>::SetValue(std::unique_ptr<T>&& new_value) {
  value_ = std::move(new_value);
}

template <class T>
int Node<T>::NodeId() const {
  return node_id_;
}

template <class T>
void RemoveParentChildEdge(Node<T>& parent, Node<T>& child) {
  child.RemoveParent(parent);
  parent.RemoveChild(child);
}

template <class T>
Node<T>::Node(int node_id, std::unique_ptr<T>&& value,
              const std::vector<int>& ancestors)
    : node_id_(node_id),
      value_(std::move(value)),
      parents_{},
      children_{},
      ancestor_node_ids_{ancestors} {
  max_used_node_ids_ = std::max(max_used_node_ids_, node_id_);
}

template <typename T>
Node<T>::Node(std::unique_ptr<T>&& value, const std::vector<int>& ancestors)
    : Node(++max_used_node_ids_, std::move(value), ancestors) {}

template <class T>
Node<T>::Node(int node_id, std::unique_ptr<T>&& value)
    : Node(node_id, std::move(value), {}) {}

template <class T>
Node<T>::Node(std::unique_ptr<T>&& value)
    : Node(++max_used_node_ids_, std::move(value), {}) {}

template <class T>
std::vector<std::shared_ptr<Node<T>>> Node<T>::Parents() const {
  std::vector<std::shared_ptr<Node<T>>> result;
  for (auto parent : parents_) {
    auto shared = parent.lock();
    CHECK(shared);
    if (!shared->IsSentinel()) {
      result.push_back(shared);
    }
  }
  return result;
}

template <class T>
const std::set<std::shared_ptr<Node<T>>>& Node<T>::Children() const {
  return children_;
}

template <class T>
void AddParentChildEdge(const std::shared_ptr<Node<T>>& parent,
                        const std::shared_ptr<Node<T>>& child) {
  CHECK(parent);
  CHECK(child);
  parent->AddChild(child);
  child->AddParent(parent);
}

}  // namespace fhelipe

#endif  // FHELIPE_NODE_H_
