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

#include "include/dag_depth_info.h"

#include <include/scaled_t_op.h>

#include "include/extended_std.h"
#include "include/t_output_c.h"
#include "include/t_rescale_c.h"

namespace fhelipe {

DagDepthInfo::DagDepthInfo(const Dag<ScaledTOp>& dag)
    : dag_(&dag),
      depth_map_(ConstructDepthMap(*dag_)),
      dag_depth_(ConstructDagDepth(depth_map_)),
      frontier_map_(ConstructFrontierMap(depth_map_, dag_depth_)),
      edges_intersecting_frontier_(ConstructEdgesIntersectingFrontier(
          *dag_, frontier_map_, depth_map_, dag_depth_)) {}

int DagDepthInfo::ConstructDagDepth(const DepthMap& depth_map) {
  int max_depth = 0;
  for (const auto& [node, depth] : depth_map) {
    if (max_depth < depth) {
      max_depth = depth;
    }
  }
  return max_depth;
}

std::shared_ptr<Node<ScaledTOp>> NodeWithHighestDepth(
    const DepthMap& depth_map,
    const std::vector<std::shared_ptr<Node<ScaledTOp>>>& nodes) {
  auto best = nodes[0];
  for (const auto& node : nodes) {
    if (depth_map.at(node.get()) > depth_map.at(best.get())) {
      best = node;
    }
  }
  return best;
}

bool IsAfterFrontier(const Node<ScaledTOp>* node,
                     const std::set<const Node<ScaledTOp>*>& frontier,
                     const DepthMap& depth_map) {
  // Special case: If current node is on frontier, then it is _before_ the
  // frontier
  if (frontier.find(node) != frontier.end()) {
    return false;
  }
  // DFS up
  for (auto curr_node = node; depth_map.at(curr_node) == depth_map.at(node);
       curr_node =
           NodeWithHighestDepth(depth_map, curr_node->Parents()).get()) {
    if (frontier.find(curr_node) != frontier.end()) {
      return true;
    }
  }
  return false;
}

EdgesIntersectingFrontierMap DagDepthInfo::ConstructEdgesIntersectingFrontier(
    const Dag<ScaledTOp>& dag, const FrontierMap& frontier_map,
    const DepthMap& depth_map, int dag_depth) {
  EdgesIntersectingFrontierMap edges_intersecting_frontier(dag_depth + 1);
  std::set<std::shared_ptr<Node<ScaledTOp>>> visited;
  for (const auto& parent_t : dag.NodesInTopologicalOrder()) {
    visited.insert(parent_t);
    int parent_depth = depth_map.at(parent_t.get());
    for (const auto& child_t : parent_t->Children()) {
      int child_depth = depth_map.at(child_t.get());
      if (child_depth - parent_depth > 1 ||
          (child_depth - parent_depth == 1 &&
           !dynamic_cast<const TRescaleC*>(&child_t->Value().GetTOp()))) {
        for (int depth = parent_depth; depth < child_depth - 1; ++depth) {
          edges_intersecting_frontier.at(depth + 1).insert(
              DirectedEdge<ScaledTOp>(parent_t, child_t));
        }
        if (IsAfterFrontier(child_t.get(), frontier_map.at(child_depth),
                            depth_map)) {
          edges_intersecting_frontier.at(child_depth)
              .insert(DirectedEdge<ScaledTOp>(parent_t, child_t));
        }
      }
    }
  }

  return edges_intersecting_frontier;
}

std::set<const Node<ScaledTOp>*> Sc(const DagDepthInfo& dag, int i, int j) {
  auto edges = Estd::set_to_vector(Estd::set_difference(
      dag.EdgesIntersectingFrontier(i), dag.EdgesIntersectingFrontier(j)));
  return Estd::vector_to_set(Estd::transform(
      edges, [](const auto& edge) { return edge.Parent().get(); }));
}

int ScCount(const DagDepthInfo& dag, int i, int j) {
  return Sc(dag, i, j).size();
}

DepthMap DagDepthInfo::ConstructDepthMap(const Dag<ScaledTOp>& dag) {
  DepthMap depth_map;
  for (const auto& node : dag.NodesInTopologicalOrder()) {
    int depth = 0;
    if (!node->Parents().empty()) {
      depth =
          Estd::max_element(Estd::values_from_keys(depth_map, node->Parents()));
      if (dynamic_cast<const TRescaleC*>(&node->Value().GetTOp())) {
        ++depth;
      }
    }
    depth_map.emplace(node.get(), depth);
  }

  return depth_map;
}

std::vector<const Node<ScaledTOp>*> FindSinkNodes(
    const std::vector<const Node<ScaledTOp>*>& nodes) {
  return Estd::filter(nodes, [&nodes](const auto* node) {
    return Estd::all_of(node->Children(), [&nodes](const auto& child) {
      return dynamic_cast<const TOutputC*>(&child->Value().GetTOp()) ||
             !Estd::contains(nodes, child.get());
    });
  });
}

std::vector<const Node<ScaledTOp>*> FindSourceNodes(
    const std::vector<const Node<ScaledTOp>*>& nodes) {
  return Estd::filter(nodes, [&nodes](const auto* node) {
    return Estd::all_of(node->Parents(), [&nodes](const auto& child) {
      return !Estd::contains(nodes, child.get());
    });
  });
}

bool AllParentsVisited(
    const std::vector<const Node<ScaledTOp>*>& nodes_to_be_considered,
    const VisitedNodes<ScaledTOp>& visited, const Node<ScaledTOp>& node) {
  return Estd::all_of(
      node.Parents(), [&nodes_to_be_considered, &visited](const auto& parent) {
        return !Estd::contains(nodes_to_be_considered, parent.get()) ||
               visited.IsVisited(*parent);
      });
}

std::vector<const Node<ScaledTOp>*> TopoSort(
    const std::vector<const Node<ScaledTOp>*>& nodes) {
  std::vector<const Node<ScaledTOp>*> result;
  std::queue<const Node<ScaledTOp>*> frontier;
  VisitedNodes<ScaledTOp> visited;
  for (const auto* sink : FindSourceNodes(nodes)) {
    frontier.push(sink);
    visited.Set(*sink);
  }

  // Topo sort
  while (!frontier.empty()) {
    const auto* vertex = frontier.front();
    frontier.pop();
    result.push_back(vertex);

    for (const auto& child : vertex->Children()) {
      if (AllParentsVisited(nodes, visited, *child) &&
          !visited.IsVisited(*child) && Estd::contains(nodes, child.get())) {
        frontier.push(child.get());
        visited.Set(*child);
      }
    }
  }

  return result;
}

std::vector<const Node<ScaledTOp>*> ReverseTopoSort(
    const std::vector<const Node<ScaledTOp>*>& nodes) {
  return Estd::reverse(TopoSort(nodes));
}

bool NoChildARescale(const Node<ScaledTOp>* node) {
  return Estd::all_of(node->Children(), [](const auto& child) {
    return !dynamic_cast<const TRescaleC*>(&child->Value().GetTOp());
  });
}

std::optional<const Node<ScaledTOp>*> FindChokepoint(
    std::vector<const Node<ScaledTOp>*> nodes) {
  nodes = ReverseTopoSort(nodes);
  auto dependencies = Estd::vector_to_set(Estd::transform(
      FindSinkNodes(nodes),
      [&nodes](const auto* x) -> int { return Estd::find_index(nodes, x); }));

  for (int idx : Estd::indices(nodes.size() - 1)) {
    if (Estd::max_element(dependencies) <= idx &&
        NoChildARescale(nodes.at(idx))) {
      return nodes.at(idx);
    }
    auto parents_at_depth = Estd::filter(
        nodes.at(idx)->Parents(),
        [&nodes](const auto& x) { return Estd::contains(nodes, x.get()); });

    std::vector<int> parents_at_depth_indices =
        Estd::transform(parents_at_depth, [&nodes](const auto& node) -> int {
          const Node<ScaledTOp>* dummy = node.get();
          return Estd::find_index(nodes, dummy);
        });

    dependencies = Estd::set_union(
        dependencies, Estd::vector_to_set(parents_at_depth_indices));
  }
  auto srcs = FindSourceNodes(nodes);
  if (srcs.size() == 1) {
    return srcs.at(0);
  }
  return std::nullopt;
}

std::vector<std::vector<const Node<ScaledTOp>*>> DivideIntoConnectedComponents(
    const std::vector<const Node<ScaledTOp>*>& nodes) {
  std::vector<std::vector<const Node<ScaledTOp>*>> result;
  std::unordered_map<const Node<ScaledTOp>*, int> components;
  for (const auto& node : TopoSort(nodes)) {
    auto ancestors = std::set<const Node<ScaledTOp>*>{node};
    int elem_count;
    do {
      elem_count = ancestors.size();
      auto new_ancestors = ancestors;
      for (const auto* node : ancestors) {
        auto curr_parents =
            Estd::filter(node->Parents(), [&nodes](const auto& parent) {
              return Estd::contains(nodes, parent.get());
            });
        new_ancestors = Estd::set_union(
            new_ancestors,
            Estd::vector_to_set(Estd::transform(
                curr_parents, [](const auto& node) -> const Node<ScaledTOp>* {
                  return node.get();
                })));
        auto curr_children =
            Estd::filter(node->Children(), [&nodes](const auto& child) {
              return Estd::contains(nodes, child.get());
            });
        new_ancestors = Estd::set_union(
            new_ancestors,
            Estd::transform(curr_children,
                            [](const auto& node) -> const Node<ScaledTOp>* {
                              return node.get();
                            }));
      }
      ancestors = new_ancestors;
    } while (elem_count != ancestors.size());

    int component_label = Estd::min_element(Estd::transform(
        ancestors, [](const auto& node) { return node->NodeId(); }));

    for (const auto& ancestor : ancestors) {
      components[ancestor] = component_label;
    }
  }

  std::unordered_map<int, std::vector<const Node<ScaledTOp>*>> reverse_map;
  for (const auto& [node, component_id] : components) {
    reverse_map[component_id].push_back(node);
  }
  for (const auto& [id, component_nodes] : reverse_map) {
    result.push_back(component_nodes);
  }
  return result;
}

std::set<const Node<ScaledTOp>*> FindChokepointsPerConnectedComponent(
    const std::vector<const Node<ScaledTOp>*>& nodes) {
  auto components = DivideIntoConnectedComponents(nodes);
  std::set<const Node<ScaledTOp>*> result;
  for (const auto& component : components) {
    auto chokepoint = FindChokepoint(component);
    if (chokepoint.has_value()) {
      result = Estd::set_union(result, std::set{chokepoint.value()});
    } else {
      result = Estd::set_union(
          result,
          Estd::vector_to_set(Estd::filter(component, [](const auto* node) {
            return dynamic_cast<const TRescaleC*>(&node->Value().GetTOp());
          })));
    }
  }
  return result;
}

FrontierMap DagDepthInfo::ConstructFrontierMap(const DepthMap& depth_map,
                                               int dag_depth) {
  FrontierMap frontier_map(dag_depth + 1);

  // Construct depth_to_nodes
  std::vector<std::vector<const Node<ScaledTOp>*>> depth_to_nodes(dag_depth +
                                                                  1);
  for (const auto& [node, depth] : depth_map) {
    depth_to_nodes.at(depth).push_back(node);
  }

  for (int curr_depth : Estd::indices(dag_depth + 1)) {
    const auto& curr_nodes = depth_to_nodes.at(curr_depth);
    frontier_map.at(curr_depth) =
        FindChokepointsPerConnectedComponent(curr_nodes);
  }
  return frontier_map;
}

bool EdgeIntersectsFrontier(const DagDepthInfo& dag, int frontier,
                            const DirectedEdge<ScaledTOp>& edge) {
  return Estd::contains(dag.EdgesIntersectingFrontier(frontier), edge);
}

}  // namespace fhelipe
