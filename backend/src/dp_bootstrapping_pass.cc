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

#include "include/dp_bootstrapping_pass.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "include/bootstrapping_pass_utils.h"
#include "include/constants.h"
#include "include/dag.h"
#include "include/dag_depth_info.h"
#include "include/debug_info.h"
#include "include/extended_std.h"
#include "include/io_utils.h"
#include "include/level.h"
#include "include/level_info.h"
#include "include/level_info_utils.h"
#include "include/leveled_t_op.h"
#include "include/node.h"
#include "include/pass_utils.h"
#include "include/program_context.h"
#include "include/scaled_t_op.h"
#include "include/t_add_cc.h"
#include "include/t_add_cp.h"
#include "include/t_add_csi.h"
#include "include/t_bootstrap_c.h"
#include "include/t_input_c.h"
#include "include/t_mul_cc.h"
#include "include/t_mul_cp.h"
#include "include/t_mul_csi.h"
#include "include/t_op.h"
#include "include/t_output_c.h"
#include "include/t_rescale_c.h"
#include "include/utils.h"
#include "latticpp/ckks/ciphertext.h"

namespace fhelipe {

namespace {

std::vector<std::vector<const Node<ScaledTOp>*>> bootstrap_at_to_shortcuts;
std::vector<std::vector<int>> bootstrap_at_to_levels;

/*
bool EdgeIntersectsBootstrappingFrontier(
    const DagDepthInfo& dag, const std::vector<int>& bootstrapping_frontiers,
    const DirectedEdge<ScaledTOp>& edge) {
  return Estd::any_of(bootstrapping_frontiers, [&dag, &edge](int frontier) {
    return EdgeIntersectsFrontier(dag, frontier, edge);
  });
}

bool RequiresBootstrappingAsShortcut(
    const std::shared_ptr<Node<ScaledTOp>> old_node,
    const std::vector<int>& bootstrapping_frontiers,
    const DagDepthInfo& dag_depth_info) {
  auto ChildEdgeIntersectsBootstrappingFrontier =
      [&dag_depth_info, &bootstrapping_frontiers,
       &old_node](const auto& child) {
        return EdgeIntersectsBootstrappingFrontier(
            dag_depth_info, bootstrapping_frontiers,
            DirectedEdge<ScaledTOp>(old_node, child));
      };

  return Estd::any_of(old_node->Children(),
                      ChildEdgeIntersectsBootstrappingFrontier);
}
*/

bool RequiresBootstrappingAsFrontier(
    const Node<ScaledTOp>& old_node,
    const std::vector<int>& bootstrapping_frontiers,
    const DagDepthInfo& dag_depth_info) {
  auto NodeOnFrontierLambda = [&dag_depth_info, &old_node](int frontier) {
    return NodeOnFrontier(dag_depth_info, frontier, old_node);
  };

  return Estd::any_of(bootstrapping_frontiers, NodeOnFrontierLambda);
}

/*
bool RequiresBootstrapping(const std::shared_ptr<Node<ScaledTOp>>& old_node,
                           const std::vector<int>& bootstrapping_frontiers,
                           const DagDepthInfo& dag_depth_info) {
  return RequiresBootstrappingAsFrontier(*old_node, bootstrapping_frontiers,
                                         dag_depth_info) ||
         RequiresBootstrappingAsShortcut(old_node, bootstrapping_frontiers,
                                         dag_depth_info);
};
*/

class DynamicProgrammingElement {
 public:
  explicit DynamicProgrammingElement(int prev_index, int dp_value)
      : prev_index_(prev_index), dp_value_(dp_value) {}

  int PreviousIndex() const { return prev_index_; }
  int DpValue() const { return dp_value_; }

 private:
  int prev_index_;
  int dp_value_;
};

class DynamicProgrammingHelper {
 public:
  typedef std::function<DynamicProgrammingElement(
      const std::vector<DynamicProgrammingElement>&, int)>
      DpFunc;

  DynamicProgrammingHelper(
      const std::vector<DynamicProgrammingElement>& init_dp_array,
      const DpFunc& dp_func)
      : dp_array_(init_dp_array), dp_func_(dp_func) {}

  std::vector<int> ComputeResult(int range_begin, int range_end);

 private:
  std::vector<int> BacktrackToProduceResult();
  std::vector<DynamicProgrammingElement> dp_array_;
  DpFunc dp_func_;
};

std::vector<int> DynamicProgrammingHelper::ComputeResult(int range_begin,
                                                         int range_end) {
  for (int i = range_begin; i < range_end; ++i) {
    dp_array_.emplace_back(dp_func_(dp_array_, i));
  }
  return BacktrackToProduceResult();
}

std::vector<int> DynamicProgrammingHelper::BacktrackToProduceResult() {
  int end_point = dp_array_.size() - 1;
  std::vector<int> result;
  std::cout << "Thinks: " << std::endl;
  for (int j = dp_array_[end_point].PreviousIndex(); j != 0;
       j = dp_array_[j].PreviousIndex()) {
    std::cout << j << " " << dp_array_[j].DpValue() << std::endl;
    result.push_back(j);
  }

  return result;
}

int NextBootstrappingDepth(const Node<ScaledTOp>* node,
                           const std::vector<int>& bootstrapping_depths,
                           const DagDepthInfo& dag_depth_info) {
  int me = dag_depth_info.NodeDepth(node);
  if (IsAfterFrontier(node,
                      dag_depth_info.Frontier(dag_depth_info.NodeDepth(node)),
                      dag_depth_info.GetDepthMap())) {
    me++;
  }
  int candidate = 1000000;
  for (int depth : bootstrapping_depths) {
    if (depth >= me && depth < candidate) {
      candidate = depth;
    }
  }
  if (candidate == 1000000) {
    LOG(FATAL);
  }
  return candidate;
}

int PreviousBootstrappingDepth(const Node<ScaledTOp>* node,
                               const std::vector<int>& bootstrapping_depths,
                               const DagDepthInfo& dag_depth_info) {
  if (IsAfterFrontier(node,
                      dag_depth_info.Frontier(dag_depth_info.NodeDepth(node)),
                      dag_depth_info.GetDepthMap())) {
    return Estd::closest_element_less_than_or_equal_to(
        bootstrapping_depths, dag_depth_info.NodeDepth(node));
  }
  if (dag_depth_info.NodeDepth(node) == 0) {
    return 0;
  }
  return Estd::closest_element_less_than_or_equal_to(
      bootstrapping_depths, dag_depth_info.NodeDepth(node) - 1);
}

int ShortcutPain(const DagDepthInfo& dag_depth_info,
                 const std::vector<DynamicProgrammingElement>& dp_array,
                 Level usable_levels, int prev, int curr,
                 const std::vector<int>& levels) {
  auto shortcuts =
      Sc(dag_depth_info, curr, std::max(0, curr - usable_levels.value()));
  int result = 0;
  std::vector<int> frontiers;
  for (int j = prev; j != 0; j = dp_array.at(j).PreviousIndex()) {
    frontiers.push_back(j);
  }
  frontiers.push_back(0);
  for (const auto* shortcut : shortcuts) {
    int closest_boy =
        PreviousBootstrappingDepth(shortcut, frontiers, dag_depth_info);
    std::cout << "closest_boy: " << closest_boy << std::endl;
    if (closest_boy == prev) {
      std::cout << "if" << std::endl;
      std::cout << "closest_boy: " << closest_boy << std::endl;
      std::cout << "next_boy: " << curr << std::endl;
      int idx =
          std::max(0, usable_levels.value() - 1 -
                          (dag_depth_info.NodeDepth(shortcut) - closest_boy));
      result += (usable_levels.value() - (levels.at(idx)));
    } else {
      std::cout << "else" << std::endl;
      int next_boy =
          NextBootstrappingDepth(shortcut, frontiers, dag_depth_info);
      std::cout << "closest_boy: " << closest_boy << std::endl;
      std::cout << "next_boy: " << next_boy << std::endl;
      int idx =
          std::max(0, usable_levels.value() - 1 -
                          (dag_depth_info.NodeDepth(shortcut) - closest_boy));
      std::cout << "idx: " << idx << std::endl;
      result += (usable_levels.value() -
                 (bootstrap_at_to_levels.at(next_boy).at(idx)));
    }
  }
  return result;
}

DynamicProgrammingElement SelectMinimum(
    const DagDepthInfo& dag_depth_info,
    const std::vector<DynamicProgrammingElement>& dp_array,
    const std::vector<DynamicProgrammingElement>& values, int curr_depth,
    const std::vector<std::vector<const Node<ScaledTOp>*>>& shortcuts,
    const std::vector<std::vector<int>>& levels) {
  int j = 0;
  int min_dp_value = 1000000;
  int min_idx = 0;
  int idx = 0;

  for (const auto& value : values) {
    if (min_dp_value > value.DpValue()) {
      min_dp_value = value.DpValue();
      j = value.PreviousIndex();
      min_idx = idx;
    } else if (min_dp_value == value.DpValue()) {
      // Breaking ties by choosing the guy with least pain
      if (ShortcutPain(dag_depth_info, dp_array, levels.size(),
                       value.PreviousIndex(), curr_depth, levels[idx]) <
          ShortcutPain(dag_depth_info, dp_array, levels.size(),
                       values[min_idx].PreviousIndex(), curr_depth,
                       levels[min_idx])) {
        min_dp_value = value.DpValue();
        j = value.PreviousIndex();
        min_idx = idx;
      }
    }
    idx++;
  }

  bootstrap_at_to_shortcuts.push_back(shortcuts[min_idx]);
  bootstrap_at_to_levels.push_back(levels[min_idx]);
  std::cout << "Levels: ";
  std::cout << "SelectMinimum: " << bootstrap_at_to_shortcuts.size() << " "
            << bootstrap_at_to_levels.size() << " "
            << bootstrap_at_to_levels.back().size() << std::endl;
  return DynamicProgrammingElement(j, min_dp_value);
}

std::vector<int> BootstrappingFrontiersAt(
    const std::vector<DynamicProgrammingElement>& dp_array, int end_depth) {
  int helper = end_depth;
  auto bootstrapping_frontiers = std::vector<int>{helper};
  while (helper > 0) {
    helper = dp_array.at(helper).PreviousIndex();
    bootstrapping_frontiers.push_back(helper);
  }
  bootstrapping_frontiers.push_back(0);

  return bootstrapping_frontiers;
}

std::vector<std::vector<std::tuple<const Node<ScaledTOp>*, int, int>>>
AllSubsets(
    std::vector<std::tuple<const Node<ScaledTOp>*, int, int>> shortcuts) {
  std::vector<std::vector<std::tuple<const Node<ScaledTOp>*, int, int>>> result;
  if (shortcuts.empty()) {
    return {{}};
  }
  auto back = shortcuts.back();
  shortcuts.pop_back();
  auto tail = AllSubsets(shortcuts);
  result = Estd::concat(
      Estd::transform(
          tail,
          [&back](const auto& vec) {
            std::vector<std::tuple<const Node<ScaledTOp>*, int, int>> vec_new(
                vec.begin(), vec.end());
            vec_new.push_back(back);
            return vec_new;
          }),
      tail);
  return result;
}

std::vector<int> ShaveLevels(std::vector<int> levels, int bottom_level,
                             int top_level) {
  int x = levels.size() - bottom_level;
  for (int i = x; i > 0; i--) {
    levels[i - 1] = std::min(levels[i - 1], bottom_level - top_level + i);
  }
  return levels;
}

std::pair<std::set<const Node<ScaledTOp>*>, std::vector<int>>
PickLargestAcceptableShortcutSubset(
    const std::set<std::tuple<const Node<ScaledTOp>*, int, int>>& shortcuts,
    int curr_width, Level usable_levels) {
  auto shortcuts_vec = Estd::set_to_vector(shortcuts);
  std::vector<std::tuple<const Node<ScaledTOp>*, int, int>> max_subset{};
  std::vector<int> max_levels = Estd::indices(1, 1 + usable_levels.value());
  for (const auto& subset : AllSubsets(shortcuts_vec)) {
    if (subset.size() <= max_subset.size()) {
      continue;
    }
    auto levels = Estd::indices(1, 1 + usable_levels.value());
    for (const auto& [node, top_level, bottom_level] : subset) {
      /*
    std::cout << "TOP: " << top_level << "; BOTTOM: " << bottom_level
              << std::endl;
              */
      levels = ShaveLevels(levels, bottom_level, top_level);
      /*
      std::cout << "Levels: ";
      WriteStream(std::cout, levels);
      std::cout << std::endl;
      */
      if (levels[levels.size() - 1 - curr_width] < 1) {
        continue;
      }
    }
    if (levels[levels.size() - 1 - curr_width] > 0 &&
        subset.size() > max_subset.size()) {
      max_subset = std::vector<std::tuple<const Node<ScaledTOp>*, int, int>>(
          subset.begin(), subset.end());
      std::cout << "Levels: ";
      WriteStream(std::cout, levels);
      max_levels = levels;
    }
  }
  std::cout << "MaxLevels: ";
  WriteStream(std::cout, max_levels);
  return std::make_pair(
      Estd::vector_to_set(Estd::transform(
          max_subset, [](const auto& tpl) { return std::get<0>(tpl); })),
      max_levels);
}

std::pair<std::set<const Node<ScaledTOp>*>, std::vector<int>>
FilterOutShortcutsThatDontRequireBootstrapping(
    const DagDepthInfo& dag_depth_info,
    const std::vector<DynamicProgrammingElement>& dp_array,
    std::set<const Node<ScaledTOp>*> shortcut_nodes, int j, int depth,
    Level usable_levels) {
  auto bootstrapping_frontiers = BootstrappingFrontiersAt(dp_array, j);
  auto shortcuts_with_info =
      Estd::transform(shortcut_nodes, [&usable_levels, &dag_depth_info,
                                       &bootstrapping_frontiers, depth,
                                       j](const auto* shortcut_parent) {
        auto shortcut_parent_depth = dag_depth_info.NodeDepth(shortcut_parent);
        auto shortcut_children =
            Estd::set_to_vector(shortcut_parent->Children());

        std::cout << "blah0" << std::endl;
        // Find pessimistic child
        auto shortcut_children_candidates = Estd::filter(
            shortcut_children, [&dag_depth_info, depth, j](const auto& child) {
              return dag_depth_info.NodeDepth(child.get()) >= j &&
                     dag_depth_info.NodeDepth(child.get()) <= depth;
            });

        std::cout << "blah1" << std::endl;
        int min_depth = 1000000;
        int idx = 0;
        int min_idx = 0;
        for (const auto& candidate : shortcut_children_candidates) {
          if (dag_depth_info.NodeDepth(candidate.get()) < min_depth) {
            min_depth = dag_depth_info.NodeDepth(candidate.get());
            min_idx = idx;
          }
          idx++;
        }
        std::cout << "blah2" << std::endl;
        auto shortcut_child = shortcut_children_candidates.at(min_idx);

        auto closest_bootstrap_to_parent = PreviousBootstrappingDepth(
            shortcut_parent, bootstrapping_frontiers, dag_depth_info);
        int closest_bootstrap_depth_bigger_than_parent = NextBootstrappingDepth(
            shortcut_parent, bootstrapping_frontiers, dag_depth_info);
        std::cout << "blah3" << std::endl;
        std::cout << "bootstrap_at_to_levels.size = "
                  << bootstrap_at_to_levels.size() << std::endl;
        std::cout << "bootstrap_at_to_levels.args = "
                  << closest_bootstrap_depth_bigger_than_parent << std::endl;
        std::cout << "bootstrap_at_to_levels[args] = ";
        WriteStream(
            std::cout,
            bootstrap_at_to_levels[closest_bootstrap_depth_bigger_than_parent]);
        std::cout << std::endl;
        std::cout << "args: "
                  << usable_levels.value() - 1 -
                         (shortcut_parent_depth - closest_bootstrap_to_parent)
                  << std::endl;
        return std::make_tuple(
            shortcut_parent,
            usable_levels.value() -
                bootstrap_at_to_levels
                    .at(closest_bootstrap_depth_bigger_than_parent)
                    .at(usable_levels.value() - 1 -
                        (shortcut_parent_depth - closest_bootstrap_to_parent)),
            dag_depth_info.NodeDepth(shortcut_child.get()) - j);
      });
  auto [largest_subset, levels] = PickLargestAcceptableShortcutSubset(
      shortcuts_with_info, depth - j, usable_levels);
  std::cout << largest_subset.size() << " / " << shortcuts_with_info.size()
            << std::endl;
  return std::make_pair(
      Estd::set_difference(
          Estd::transform(shortcuts_with_info,
                          [](const auto& tpl) { return std::get<0>(tpl); }),
          largest_subset),
      levels);
}

DynamicProgrammingElement DpBootstrappingCostFunction(
    const DagDepthInfo& dag_depth_info, const Level& usable_levels,
    const std::vector<DynamicProgrammingElement>& dp_array, int depth) {
  if (depth < usable_levels.value()) {
    return DynamicProgrammingElement{0, 0};
  }
  std::vector<DynamicProgrammingElement> candidates;
  std::vector<std::vector<const Node<ScaledTOp>*>> candidate_shortcuts;
  std::vector<std::vector<int>> candidate_levels;
  for (int j = std::max(depth - usable_levels.value() + 1, 0); j < depth; ++j) {
    const auto& nodes_at_frontier = dag_depth_info.Frontier(j);
    int ciphertext_count_at_frontier = CiphertextCount(nodes_at_frontier);
    auto old_shortcut_nodes = Sc(dag_depth_info, j, depth);
    auto [shortcut_nodes, levels] =
        FilterOutShortcutsThatDontRequireBootstrapping(dag_depth_info, dp_array,
                                                       old_shortcut_nodes, j,
                                                       depth, usable_levels);

    int shortcut_ciphertext_count = CiphertextCount(shortcut_nodes);
    std::cout << j << " --> " << depth << ": " << shortcut_ciphertext_count
              << std::endl;

    int dp_value = dp_array.at(j).DpValue() + ciphertext_count_at_frontier +
                   shortcut_ciphertext_count;
    candidates.emplace_back(DynamicProgrammingElement(j, dp_value));
    candidate_shortcuts.push_back(Estd::set_to_vector(shortcut_nodes));
    candidate_levels.push_back(levels);
  }

  return SelectMinimum(dag_depth_info, dp_array, candidates, depth,
                       candidate_shortcuts, candidate_levels);
}

std::vector<int> GetBootstrappingFrontiers(const DagDepthInfo& dag_depth_info,
                                           const Level& usable_levels) {
  // nsamar: dp[i] represents the "minimum" number of bootstraps (as far as
  // the DP algorithm knows) to validly run the program until depth i
  std::vector<DynamicProgrammingElement> dp_initial{
      DynamicProgrammingElement{0, 0}};

  auto dp_function = [&dag_depth_info, &usable_levels](
                         const std::vector<DynamicProgrammingElement>& dp_array,
                         int depth) {
    return DpBootstrappingCostFunction(dag_depth_info, usable_levels, dp_array,
                                       depth);
  };

  return DynamicProgrammingHelper(dp_initial, dp_function)
      .ComputeResult(1, dag_depth_info.DagDepth() + 1);
}

void RemoveBootstraps(const LevelingPassInput& dag) {
  for (auto& node :
       CollectNodesByTypeInTopologicalOrder<ScaledTOp, TBootstrapC>(dag)) {
    RemoveNode(*node);
  }
}

std::shared_ptr<Node<LeveledTOp>> BuildNewNode(
    Dag<LeveledTOp>& dag,
    std::vector<std::shared_ptr<Node<LeveledTOp>>>& parents,
    const std::shared_ptr<Node<ScaledTOp>>& old_node,
    const Level& usable_levels, bool bootstrap_after, bool is_shortcut,
    int node_depth) {
  std::cout << "is_shortcut: " << is_shortcut << std::endl;
  std::cout << "bootstrap_after: " << bootstrap_after << std::endl;
  std::cout << "node_depth: " << node_depth << std::endl;
  std::cout << "node: " << std::endl;
  WriteStream(std::cout, old_node->Value());
  std::cout << std::endl;
  std::cout << "node id: " << old_node->NodeId() << std::endl;
  std::cout << "node parents: ";
  WriteStream(std::cout,
              Estd::transform(old_node->Parents(), [](const auto& parent) {
                return parent->NodeId();
              }));
  std::cout << std::endl;

  auto parents_level_info = ExtractLevelInfos(parents);
  auto new_node_level_info =
      NodeLevelInfo(*old_node, parents_level_info, usable_levels);

  auto leveled_t_op = std::make_unique<LeveledTOp>(
      old_node->Value().GetTOp().CloneUniq(), new_node_level_info, node_depth);
  auto new_node =
      dag.AddNode(std::move(leveled_t_op), parents, {old_node->NodeId()});

  CHECK(new_node->Value().Level() >= kMinLevel);

  if (bootstrap_after) {
    auto boot_node = std::make_unique<TBootstrapC>(
        new_node->Value().GetTOp().OutputLayout(), usable_levels, is_shortcut);
    auto leveled_t_op = std::make_unique<LeveledTOp>(
        std::move(boot_node),
        LevelInfo{usable_levels, old_node->Value().LogScale()});
    new_node = CreateChild(std::move(leveled_t_op), {new_node});
  }

  return new_node;
}

}  // namespace

LevelingPassOutput DpBootstrappingPass::DoPass(
    const LevelingPassInput& in_dag) {
  RemoveBootstraps(in_dag);
  DagDepthInfo dag_depth_info(in_dag);

  for (int idx : Estd::indices(context_.UsableLevels().value())) {
    (void)idx;
    bootstrap_at_to_shortcuts.push_back({});
    bootstrap_at_to_levels.push_back(
        Estd::indices(1, 1 + context_.UsableLevels().value()));
  }
  std::vector<int> bootstrapping_frontiers =
      GetBootstrappingFrontiers(dag_depth_info, context_.UsableLevels());

  // Collect shortcuts
  std::vector<const Node<ScaledTOp>*> all_shortcuts;
  for (const auto& node :
       bootstrap_at_to_shortcuts.at(bootstrap_at_to_shortcuts.size() - 1)) {
    all_shortcuts.push_back(node);
  }
  for (int frontier : bootstrapping_frontiers) {
    for (const auto& node : bootstrap_at_to_shortcuts.at(frontier)) {
      all_shortcuts.push_back(node);
    }
  }

  std::cout << "Total shortcuts: " << all_shortcuts.size() << std::endl;

  Dag<LeveledTOp> out_dag;
  std::unordered_map<const Node<ScaledTOp>*, std::shared_ptr<Node<LeveledTOp>>>
      old_to_new_nodes;
  int total = in_dag.NodesInTopologicalOrder().size();
  int count = 0;
  for (const auto& old_node : in_dag.NodesInTopologicalOrder()) {
    std::cout << count++ << " / " << total << std::endl;
    auto parents = ExtractParents(old_to_new_nodes, *old_node);

    bool is_shortcut = Estd::contains(all_shortcuts, old_node.get());
    bool bootstrap_after =
        is_shortcut || RequiresBootstrappingAsFrontier(
                           *old_node, bootstrapping_frontiers, dag_depth_info);
    auto new_node = BuildNewNode(
        out_dag, parents, old_node, context_.UsableLevels(), bootstrap_after,
        is_shortcut, dag_depth_info.NodeDepth(old_node.get()));
    old_to_new_nodes.emplace(old_node.get(), new_node);
  }

  return out_dag;
}

}  // namespace fhelipe
