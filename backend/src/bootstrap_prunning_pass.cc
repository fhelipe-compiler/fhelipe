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

#include "include/bootstrap_prunning_pass.h"

#include "include/t_bootstrap_c.h"
#include "include/t_rescale_c.h"

namespace fhelipe {

namespace {

// Assumes source dag does not change (or at least the mapped nodes and their
// immediate neighbors)
template <class T>
class HypotheticalDag {
 public:
  std::shared_ptr<Node<LeveledTOp>> Mapping(Node<T>& node) {
    return map_.at(&node);
  }

  std::shared_ptr<Node<T>> NewNode(Node<T>& node) {
    AddMapping(node, CloneUniqWithSameAncestors(node));
    return Mapping(node);
  }

  std::shared_ptr<Node<T>> MappingCreateIfNeeded(Node<T>& node) {
    if (Estd::contains_key(map_, &node)) {
      return map_.at(&node);
    }
    return NewNode(node);
  }

  // Don't include the mapped node when Commit()ing
  void RemoveMapping(const std::shared_ptr<Node<T>>& node) {
    auto mapped_node = map_.at(node.get());
    dont_commit_but_also_dont_garbage_collect_.push_back(mapped_node);
    map_.erase(node.get());
  }

  bool Contains(const Node<T>& node) const { return Estd::contains_key(&node); }

  void AddMapping(Node<T>& node, const std::shared_ptr<Node<T>>& new_node) {
    CHECK(!Estd::contains_key(map_, &node));
    map_.emplace(&node, new_node);
    for (const auto& parent : node.Parents()) {
      if (Estd::contains_key(map_, parent.get())) {
        // Need to check if parent.get() still contains child, because parent
        // may appear twice in node's Parents()
        if (map_.at(parent.get())->ContainsChild(node)) {
          // Replace parent's node child with the newly created node
          map_.at(parent.get())->RemoveChild(node);
        }
        AddParentChildEdge(map_.at(parent.get()), map_.at(&node));
      } else {
        // If parent is not mapped, set the parent to the non-mapped parent
        // (but don't let the non-mapped parent know about map_.at(&node), cuz
        // this node is only hypothetical
        map_.at(&node)->AddParent(parent);
      }
    }

    // Same for children as for the parents
    for (const auto& child : node.Children()) {
      if (Estd::contains_key(map_, child.get())) {
        AddParentChildEdge(map_.at(&node), map_.at(child.get()));
        map_.at(child.get())->RemoveParent(node);
      } else {
        map_.at(&node)->AddChild(child);
      }
    }
  }

  void Commit() {
    for (auto& [old_node, new_node] : map_) {
      old_node->SetValue(new_node->Value().CloneUniq());
    }
  }

 private:
  std::unordered_map<Node<T>*, std::shared_ptr<Node<T>>> map_;
  std::vector<std::shared_ptr<Node<T>>>
      dont_commit_but_also_dont_garbage_collect_;
};

std::optional<LevelInfo> NodeLevelInfo(const Node<LeveledTOp>& node_t) {
  const auto& node = node_t.Value();
  CHECK(!dynamic_cast<const TInputC*>(&node.GetTOp()));
  if (const auto* t_bootstrap_c =
          dynamic_cast<const TBootstrapC*>(&node.GetTOp())) {
    return node.GetLevelInfo();
  }
  if (const auto* t_rescale_c =
          dynamic_cast<const TRescaleC*>(&node.GetTOp())) {
    if (node_t.Parents().at(0)->Value().Level() <= Level(1)) {
      // Level too low
      return std::nullopt;
    }
    return LevelInfo{node_t.Parents().at(0)->Value().Level() - Level(1),
                     node.LogScale()};
  }
  int min_level = Estd::min_element(Estd::transform(
      node_t.Parents(),
      [](const auto& parent) { return parent->Value().Level().value(); }));
  return LevelInfo{min_level, node.LogScale()};
}

bool EnoughLevels(Node<LeveledTOp>& node) {
  return NodeLevelInfo(node).has_value();
}

void UpdateLevelInfo(Node<LeveledTOp>& node) {
  CHECK(NodeLevelInfo(node).has_value());
  auto new_level_info = NodeLevelInfo(node).value();
  node.SetValue(std::make_unique<LeveledTOp>(node.Value().GetTOp().CloneUniq(),
                                             new_level_info));
}

bool IsMappedLevelEqual(HypotheticalDag<LeveledTOp>& hypothetical_dag,
                        Node<LeveledTOp>& node) {
  return node.Value().Level() ==
         hypothetical_dag.Mapping(node)->Value().Level();
}

template <typename T>
class DfsWalker {
 public:
  explicit DfsWalker(const std::vector<std::shared_ptr<Node<T>>>& dfs_frontier,
                     const std::function<bool(Node<T>&)>& should_add_children)
      : dfs_frontier_(dfs_frontier),
        should_add_children_(should_add_children) {}

  std::shared_ptr<Node<T>> Begin() {
    if (dfs_frontier_.empty()) {
      return std::shared_ptr<Node<T>>{};
    }
    return dfs_frontier_.back();
  }

  std::shared_ptr<Node<T>> Next();

 private:
  std::vector<std::shared_ptr<Node<T>>> dfs_frontier_;
  std::function<bool(Node<T>&)> should_add_children_;
};

template <typename T>
std::shared_ptr<Node<T>> DfsWalker<T>::Next() {
  CHECK(!dfs_frontier_.empty());

  auto prev_node = dfs_frontier_.back();
  dfs_frontier_.pop_back();

  if (should_add_children_(*prev_node)) {
    for (const auto& child : prev_node->Children()) {
      dfs_frontier_.push_back(child);
    }
  }

  return dfs_frontier_.empty() ? std::shared_ptr<Node<T>>{}
                               : dfs_frontier_.back();
}

void PruneBootstrapNode(
    const std::shared_ptr<Node<LeveledTOp>>& bootstrap_node) {
  CHECK(dynamic_cast<const TBootstrapC*>(&bootstrap_node->Value().GetTOp()));

  HypotheticalDag<LeveledTOp> dag_if_prunned;
  // Assume bootstrap doesn't exist by assigning its parent's LevelInfo to it
  dag_if_prunned.AddMapping(*bootstrap_node,
                            CloneUniqWithSameAncestors(*bootstrap_node));
  dag_if_prunned.Mapping(*bootstrap_node)
      ->Value()
      .SetLevelInfo(bootstrap_node->Parents().at(0)->Value().GetLevelInfo());

  DfsWalker<LeveledTOp> dfs(
      {bootstrap_node},
      [&dag_if_prunned, &bootstrap_node](Node<LeveledTOp>& node) {
        return &node == bootstrap_node.get() ||
               !IsMappedLevelEqual(dag_if_prunned, node);
      });

  for (auto node = dfs.Begin(); node != nullptr; node = dfs.Next()) {
    auto mapped_node = dag_if_prunned.MappingCreateIfNeeded(*node);
    if (!EnoughLevels(*mapped_node)) {
      // Cannot prune;
      return;
    }
    UpdateLevelInfo(*mapped_node);
  }

  // Prune
  dag_if_prunned.RemoveMapping(bootstrap_node);
  RemoveNode(*bootstrap_node);
  dag_if_prunned.Commit();
}

void PruneRedundantBootstraps(
    const std::vector<std::shared_ptr<Node<LeveledTOp>>>& bootstrapped_nodes) {
  // Prune shortcuts first
  for (const auto& boot_node : bootstrapped_nodes) {
    if (dynamic_cast<const TBootstrapC*>(&boot_node->Value().GetTOp())
            ->IsShortcut()
            .has_value() &&
        dynamic_cast<const TBootstrapC*>(&boot_node->Value().GetTOp())
            ->IsShortcut()
            .value()) {
      PruneBootstrapNode(boot_node);
    }
  }
  // Prune non-shortcuts second
  for (const auto& boot_node : bootstrapped_nodes) {
    if (dynamic_cast<const TBootstrapC*>(&boot_node->Value().GetTOp())
            ->IsShortcut()
            .has_value() &&
        !dynamic_cast<const TBootstrapC*>(&boot_node->Value().GetTOp())
             ->IsShortcut()
             .value()) {
      PruneBootstrapNode(boot_node);
    }
  }

  // Prune everything else
  for (const auto& boot_node : bootstrapped_nodes) {
    if (!dynamic_cast<const TBootstrapC*>(&boot_node->Value().GetTOp())
             ->IsShortcut()
             .has_value()) {
      PruneBootstrapNode(boot_node);
    }
  }
}

}  // namespace

LevelingOptimizerOutput BootstrapPrunningPass::DoPass(
    const LevelingOptimizerInput& in_dag) {
  auto out_dag = CloneFromAncestor(in_dag);
  PruneRedundantBootstraps(
      CollectNodesByTypeInTopologicalOrder<LeveledTOp, TBootstrapC>(out_dag));

  return out_dag;
}

}  // namespace fhelipe
