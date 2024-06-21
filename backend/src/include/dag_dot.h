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

#ifndef FHELIPE_DAG_DOT_H_
#define FHELIPE_DAG_DOT_H_

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

#include "bootstrap_c.h"
#include "dag.h"
#include "dag_io.h"
#include "debug_info.h"
#include "filesystem_utils.h"
#include "io_utils.h"
#include "leveled_t_op.h"
#include "scaled_t_op.h"
#include "t_bootstrap_c.h"
#include "t_chet_repack_c.h"
#include "t_layout_conversion_c.h"
#include "t_op.h"
#include "t_op_embrio.h"
#include "utils.h"

namespace fhelipe {

class Rgb;

template <class T>
using LabelFunc = std::function<std::string(const Node<T>& node)>;

template <class T>
using ColorFunc = std::function<Rgb(const Node<T>& node)>;

class Rgb {
 public:
  Rgb(int red, int green, int blue) : red_(red), green_(green), blue_(blue) {
    CHECK(red_ >= 0 && red_ < 256);
    CHECK(green_ >= 0 && green_ < 256);
    CHECK(blue_ >= 0 && blue_ < 256);
  }

  int Red() const { return red_; }
  int Green() const { return green_; }
  int Blue() const { return blue_; }

 private:
  int red_;
  int green_;
  int blue_;
};

template <typename T>
Rgb DefaultColorFunc(const Node<T>& node) {
  (void)node;
  return Rgb(255, 255, 255);
}

template <>
Rgb DefaultColorFunc<LeveledTOp>(const Node<LeveledTOp>& node) {
  if (const auto* t_bootstrap_c =
          dynamic_cast<const TBootstrapC*>(&node.Value().GetTOp())) {
    return Rgb(255, 0, 0);
  } else {
    if (!node.Value().Depth().has_value()) {
      return Rgb(255, 255, 255);
    }
    return {255, 255, 255};
    // Some high-contrast colors
    switch (node.Value().Depth().value() % 5) {
      case 0:
        return {242, 202, 25};
      case 1:
        return {255, 0, 189};
      case 2:
        return {0, 87, 233};
      case 3:
        return {135, 233, 17};
      case 4:
        return {255, 24, 69};
      default:
        LOG(FATAL);
    }
  }
}

template <>
Rgb DefaultColorFunc<TOp>(const Node<TOp>& node) {
  const TChetRepackC* repack = dynamic_cast<const TChetRepackC*>(&node.Value());
  if (dynamic_cast<const TLayoutConversionC*>(&node.Value()) ||
      (repack && repack->InputLayout() != repack->OutputLayout())) {
    return Rgb(255, 0, 0);  // Red
  } else {
    return Rgb(255, 255, 255);
  }
}

template <>
Rgb DefaultColorFunc<CtOp>(const Node<CtOp>& node) {
  if (dynamic_cast<const BootstrapC*>(&node.Value())) {
    return Rgb(255, 0, 0);  // Red
  } else {
    return Rgb(255, 255, 255);
  }
}

template <typename T>
std::string DefaultLabelFunc(const Node<T>& node) = delete;

template <>
inline std::string DefaultLabelFunc<CtOp>(const Node<CtOp>& node) {
  return node.Value().TypeName() + "_" + std::to_string(node.NodeId());
}

template <>
inline std::string DefaultLabelFunc<TOpEmbrio>(const Node<TOpEmbrio>& node) {
  return node.Value().TypeName() + "_" + std::to_string(node.NodeId());
}

template <>
inline std::string DefaultLabelFunc<TOp>(const Node<TOp>& node) {
  auto result = node.Value().TypeName() + "_" + std::to_string(node.NodeId()) +
                "\\n" + DagLabel(node.Value().OutputLayout());
  const TChetRepackC* repack = dynamic_cast<const TChetRepackC*>(&node.Value());
  if ((repack != nullptr) && repack->InputLayout() != repack->OutputLayout()) {
    result += "\\nbad repack";
  }
  return result;
}

template <>
inline std::string DefaultLabelFunc<ScaledTOp>(const Node<ScaledTOp>& node) {
  return node.Value().GetTOp().TypeName() + "_" + std::to_string(node.NodeId());
}

template <>
inline std::string DefaultLabelFunc<LeveledTOp>(const Node<LeveledTOp>& node) {
  auto result =
      node.Value().GetTOp().TypeName() + "_" + std::to_string(node.NodeId()) +
      "\\n" + "depth: " + ToString(node.Value().Depth()) + std::string("\\n") +
      "level: _" + ToString(node.Value().GetLevelInfo().Level()) +
      std::string("_\\n") +
      "weight: " + ToString(node.Value().GetTOp().OutputLayout().TotalChunks());
  if (const auto* t_bootstrap_c =
          dynamic_cast<const TBootstrapC*>(&node.Value().GetTOp())) {
    result += (std::string("\\n\\n\\n") +
               "is_shortcut: " + ToString(t_bootstrap_c->IsShortcut()));
  }
  return result;
}

std::string ColorToHexString(const Rgb& color) {
  std::stringstream ss;
  ss << "\"#";
  ss << std::hex << std::setw(2) << std::setfill('0') << color.Red();
  ss << std::hex << std::setw(2) << std::setfill('0') << color.Green();
  ss << std::hex << std::setw(2) << std::setfill('0') << color.Blue();
  ss << "\"";
  return ss.str();
}

inline std::string DotString(const Rgb& color) {
  return "[fillcolor=" + ColorToHexString(color) + ", style=filled];";
}

inline std::string DotString(const std::string& label) {
  return "[label=\"" + label + "\"]";
}

inline std::string DotDirectedEdgeString(int parent, int child) {
  return std::to_string(parent) + " -> " + std::to_string(child);
}

inline void InitDigraph(std::ostream& stream) {
  stream << "digraph graph_name\n{\n";
  stream << "fontname = \"Monospace\"\n";
  stream << "node [fontname = \"Monospace\"]\n";
}

inline void FinilizeDigraph(std::ostream& stream) { stream << "}\n"; }

template <class DagT, class T>
void AddNodesToDigraph(std::ostream& stream, const DagT& dag,
                       const LabelFunc<T>& label_func,
                       const ColorFunc<T>& color_func) {
  for (const auto& node : dag.NodesInTopologicalOrder()) {
    stream << node->NodeId() << ' ' << DotString(label_func(*node))
           << DotString(color_func(*node)) << '\n';
    for (const auto& parent : node->Parents()) {
      stream << DotDirectedEdgeString(parent->NodeId(), node->NodeId()) << '\n';
    }
  }
}

namespace detail {

template <class T>
std::string TypeName(const T& node) {
  return node.GetTOp().TypeName();
}

template <>
inline std::string TypeName<TOp>(const TOp& node) {
  return node.TypeName();
}

template <>
inline std::string TypeName<TOpEmbrio>(const TOpEmbrio& node) {
  return node.TypeName();
}

template <class SrcT>
std::string ClusterLabel(const std::vector<const Node<SrcT>*>& nodes) {
  std::string result;
  for (const auto& node : nodes) {
    result +=
        (TypeName(node->Value()) + "_" + std::to_string(node->NodeId()) + ", ");
  }
  return result;
}

}  // namespace detail

template <class SrcDagT, class DestDagT, class SrcT, class DestT>
void AddGroupsToDigraph(
    std::ostream& stream,
    const DebugInfo<SrcDagT, DestDagT, SrcT, DestT>& debug_info) {
  for (const auto& [srcs, dests] : debug_info.Mappings()) {
    if (srcs.empty()) {
      continue;
    }
    stream << "subgraph cluster_" << Estd::min_element(srcs) << " {\n";
    stream << "label=\"";
    stream << detail::ClusterLabel(srcs);
    stream << "\";\n";
    stream << "rank = same; ";
    for (const auto& node : dests) {
      stream << node->NodeId() << "; ";
    }
    stream << "\n}\n";
  }
}

template <class T>
void DagToDotfile(std::ostream& stream, const Dag<T>& dag,
                  const LabelFunc<T>& label_func,
                  const ColorFunc<T>& color_func) {
  InitDigraph(stream);
  AddNodesToDigraph<T>(stream, dag, label_func, color_func);
  FinilizeDigraph(stream);
}

template <class SrcDagT, class DestDagT, class SrcT, class DestT>
void DagToDotfileWithGroups(
    std::ostream& stream,
    const DebugInfo<SrcDagT, DestDagT, SrcT, DestT>& debug_info,
    const LabelFunc<DestT>& label_func = DefaultLabelFunc<DestT>,
    const ColorFunc<DestT>& color_func = DefaultColorFunc<DestT>) {
  InitDigraph(stream);
  AddNodesToDigraph<DestDagT, DestT>(stream, *debug_info.DestinationDag(),
                                     label_func, color_func);
  AddGroupsToDigraph<SrcDagT, DestDagT, SrcT, DestT>(stream, debug_info);
  FinilizeDigraph(stream);
}

}  // namespace fhelipe

#endif  // FHELIPE_DAG_DOT_H_
