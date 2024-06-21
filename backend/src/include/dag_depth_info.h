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

#ifndef FHELIPE_DAG_DEPTH_INFO_H_
#define FHELIPE_DAG_DEPTH_INFO_H_

#include <memory>

#include "dag.h"
#include "scaled_t_op.h"

namespace fhelipe {

template <class T>
class DirectedEdge {
 public:
  DirectedEdge(const std::shared_ptr<Node<T>>& parent,
               const std::shared_ptr<Node<T>>& child)
      : parent_(parent), child_(child) {}

  const std::shared_ptr<const Node<T>>& Parent() const { return parent_; }
  const std::shared_ptr<const Node<T>>& Child() const { return child_; }

 private:
  std::shared_ptr<const Node<T>> parent_;
  std::shared_ptr<const Node<T>> child_;
};

template <class T>
bool operator<(const DirectedEdge<T>& lhs, const DirectedEdge<T>& rhs) {
  // Lexicographic
  return lhs.Parent() < rhs.Parent() ||
         (lhs.Parent() == rhs.Parent() && lhs.Child() < rhs.Child());
}

template <class T>
bool operator==(const DirectedEdge<T>& lhs, const DirectedEdge<T>& rhs) {
  return lhs.Parent() == rhs.Parent() && lhs.Child() == rhs.Child();
}

typedef std::unordered_map<const Node<ScaledTOp>*, int> DepthMap;
typedef std::vector<std::set<const Node<ScaledTOp>*>> FrontierMap;
typedef std::vector<std::set<DirectedEdge<ScaledTOp>>>
    EdgesIntersectingFrontierMap;

class DagDepthInfo {
 public:
  explicit DagDepthInfo(const Dag<ScaledTOp>& dag);
  int NodeDepth(const Node<ScaledTOp>* node) const {
    return depth_map_.at(node);
  }
  int DagDepth() const { return dag_depth_; }
  // TODO(nsamar): Implement MinCutAtDepth()
  std::set<const Node<ScaledTOp>*> Frontier(int depth) const {
    return frontier_map_.at(depth);
  }
  std::set<DirectedEdge<ScaledTOp>> EdgesIntersectingFrontier(int depth) const {
    return edges_intersecting_frontier_.at(depth);
  }
  int FrontierSize(int depth) const { return frontier_map_.at(depth).size(); }
  const DepthMap& GetDepthMap() const { return depth_map_; }

 private:
  static FrontierMap ConstructFrontierMap(const DepthMap& depth_map,
                                          int dag_depth);
  static DepthMap ConstructDepthMap(const Dag<ScaledTOp>& dag);
  static int ConstructDagDepth(const DepthMap& depth_map);
  static EdgesIntersectingFrontierMap ConstructEdgesIntersectingFrontier(
      const Dag<ScaledTOp>& dag, const FrontierMap& frontier_map,
      const DepthMap& depth_map, int dag_depth);

  const Dag<ScaledTOp>* dag_;
  DepthMap depth_map_;
  int dag_depth_;
  FrontierMap frontier_map_;
  EdgesIntersectingFrontierMap edges_intersecting_frontier_;
};

// TODO(nsamar): Make better name
std::set<const Node<ScaledTOp>*> Sc(const DagDepthInfo& dag, int i, int j);
int ScCount(const DagDepthInfo& dag, int i, int j);

bool EdgeIntersectsFrontier(const DagDepthInfo& dag, int frontier,
                            const DirectedEdge<ScaledTOp>& edge);
inline bool NodeOnFrontier(const DagDepthInfo& dag, int frontier,
                           const Node<ScaledTOp>& node) {
  return Estd::contains(dag.Frontier(frontier), &node);
}

bool IsAfterFrontier(const Node<ScaledTOp>* node,
                     const std::set<const Node<ScaledTOp>*>& frontier,
                     const DepthMap& depth_map);

}  // namespace fhelipe

#endif  // FHELIPE_DAG_DEPTH_INFO_H_
