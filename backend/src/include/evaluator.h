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

#ifndef FHELIPE_EVALUATOR_H_
#define FHELIPE_EVALUATOR_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "add_cc.h"
#include "add_cp.h"
#include "add_cs.h"
#include "bootstrap_c.h"
#include "ct_op_visitor.h"
#include "ct_program.h"
#include "include/extended_std.h"
#include "include/level_info.h"
#include "input_c.h"
#include "io_manager.h"
#include "io_spec.h"
#include "mul_cc.h"
#include "mul_cp.h"
#include "mul_cs.h"
#include "output_c.h"
#include "rescale_c.h"
#include "rotate_c.h"

namespace fhelipe {

namespace {

inline void CheckLevelInfoCloseEnough(const LevelInfo& lhs,
                                      const LevelInfo& rhs) {
  // TODO(nsamar): THIS NEEDS TO BE UNCOMMENTED! ONLY USED TO GET BITCHOPPER TO
  // WORK (CUZ IT DOESNT RECOGNIZE SCALES)
  /*
  CHECK(lhs.Level() == rhs.Level())
    << lhs.Level().value() << " " << rhs.Level().value();
    */
  /*
  CHECK(std::abs(lhs.LogScale() - rhs.LogScale()) < 20)
      << lhs.LogScale().value() << " " << rhs.LogScale().value();
      */
}

}  // namespace

template <class CtType>
class Evaluator : public CtOpVisitor {
 public:
  static std::unique_ptr<Dictionary<CtType>> Evaluate(
      const IoManager<CtType>& io_manager,
      const ct_program::CtProgram& ct_program,
      const Dictionary<CtType>& ct_output_chunks);

  void Visit(const Node<CtOp>& node) final;

 private:
  Evaluator(const IoManager<CtType>& io_manager,
            const ct_program::CtProgram& ct_program,
            const Dictionary<CtType>& ct_output_chunks);
  CtType VisitImpl(const Node<CtOp>& node);
  CtType VisitImpl(const InputC& node, const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const OutputC& node,
                   const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const AddCC& node, const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const AddCP& node, const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const AddCS& node, const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const MulCC& node, const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const RescaleC& node,
                   const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const MulCP& node, const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const MulCS& node, const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const RotateC& node,
                   const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const BootstrapC& node,
                   const std::vector<const CtOp*>& parents);
  CtType VisitImpl(const ZeroC& node, const std::vector<const CtOp*>& parents);
  void UpdateParentReferenceCounts(const Node<CtOp>& node);
  void InitializeReferenceCounts();

  std::unique_ptr<Dictionary<CtType>> Outputs() const {
    return outputs_->CloneUniq();
  }

  const IoManager<CtType>& io_manager_;
  const ct_program::CtProgram& ct_program_;
  std::unordered_map<const CtOp*, CtType> node_map_;
  std::unordered_map<const CtOp*, int> reference_count_;
  std::unique_ptr<Dictionary<CtType>> outputs_;
};

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const Node<CtOp>& node_t) {
  auto parents = Estd::transform(
      node_t.Parents(),
      [](const auto& ptr) -> const CtOp* { return &ptr->Value(); });

  const CtOp& node = node_t.Value();
  if (const auto* ptr = dynamic_cast<const InputC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const OutputC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const AddCC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const AddCP*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const AddCS*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const MulCC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const MulCP*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const MulCS*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const RotateC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const BootstrapC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const RescaleC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  if (const auto* ptr = dynamic_cast<const ZeroC*>(&node)) {
    return VisitImpl(*ptr, parents);
  }
  LOG(FATAL) << "Unrecognized CtOp type";
}

template <class CtType>
void Evaluator<CtType>::UpdateParentReferenceCounts(const Node<CtOp>& node) {
  for (const auto& parent : Estd::vector_to_set(node.Parents())) {
    reference_count_.at(&parent->Value())--;
    if (reference_count_.at(&parent->Value()) == 0) {
      CHECK(Estd::contains_key(node_map_, &parent->Value()));
      node_map_.erase(&parent->Value());
    }
    CHECK(!parent || reference_count_.at(&parent->Value()) >= 0);
  }
}

template <class CtType>
void Evaluator<CtType>::Visit(const Node<CtOp>& node) {
  WriteStream(std::cout, node.Value());
  std::cout << std::endl;
  const CtType ct = VisitImpl(node);
  node_map_.emplace(&node.Value(), ct);
  if (!node_map_.empty()) {
    CHECK(node_map_.begin()->second.GetChunkSize() == ct.GetChunkSize());
  }
  // Check levels, except if the node is an output node, and it is equal to
  // the zero chunk (which is always at level usable_level)
  if (!dynamic_cast<const InputC*>(&node.Value()) &&
      !dynamic_cast<const BootstrapC*>(&node.Value()) &&
      !(dynamic_cast<const OutputC*>(&node.Value()) &&
        node.Parents().at(0)->HoldsNothing())) {
    CheckLevelInfoCloseEnough(node.Value().GetLevelInfo(), ct.GetLevelInfo());
  }
  UpdateParentReferenceCounts(node);
}

template <class CtType>
void Evaluator<CtType>::InitializeReferenceCounts() {
  for (const auto& node : ct_program_.NodesInTopologicalOrder()) {
    reference_count_.emplace(&node->Value(), node->Children().size());
  }
}

template <class CtType>
Evaluator<CtType>::Evaluator(const IoManager<CtType>& io_manager,
                             const ct_program::CtProgram& ct_program,
                             const Dictionary<CtType>& ct_output_chunks)
    : io_manager_(io_manager),
      ct_program_(ct_program),
      node_map_(),
      outputs_(ct_output_chunks.CloneUniq()) {
  InitializeReferenceCounts();
}

template <class CtType>
std::unique_ptr<Dictionary<CtType>> Evaluator<CtType>::Evaluate(
    const IoManager<CtType>& io_manager,
    const ct_program::CtProgram& ct_program,
    const Dictionary<CtType>& ct_output_chunks) {
  Evaluator ct_eval(io_manager, ct_program, ct_output_chunks);
  int linum = 0;
  auto nodes = ct_program.NodesInTopologicalOrder();
  for (const auto& node : nodes) {
    CHECK(!node->HoldsNothing());
    LOG(INFO) << ++linum << " / " << nodes.size() << " " << node->NodeId()
              << " " << ToString(node->Value());
    ct_eval.Visit(*node);
  }
  return ct_eval.Outputs();
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const InputC& node,
                                    const std::vector<const CtOp*>& parents) {
  CHECK(parents.empty());
  return io_manager_.Ct(node.GetIoSpec());
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const BootstrapC& node,
                                    const std::vector<const CtOp*>& parents) {
  (void)node;
  return node_map_.at(parents.at(0))
      .BootstrapC(ct_program_.GetProgramContext().UsableLevels());
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const RescaleC& node,
                                    const std::vector<const CtOp*>& parents) {
  (void)node;
  return node_map_.at(parents.at(0))
      .RescaleC(ct_program_.GetProgramContext().LogScale());
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const OutputC& node,
                                    const std::vector<const CtOp*>& parents) {
  // CHECK(!parents.at(0) || node.GetLevelInfo() ==
  // parents.at(0)->GetLevelInfo());
  outputs_->Record(ToFilename(IoSpec(node.GetIoSpec())),
                   node_map_.at(parents.at(0)));
  return node_map_.at(parents.at(0));
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const AddCC& node,
                                    const std::vector<const CtOp*>& parents) {
  (void)node;
  const auto& ct0 = node_map_.at(parents.at(0));
  const auto& ct1 = node_map_.at(parents.at(1));
  return ct0.AddCC(ct1);
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const AddCP& node,
                                    const std::vector<const CtOp*>& parents) {
  (void)node;
  CHECK(parents.at(0));
  const auto& parent_ct = node_map_.at(parents.at(0));
  auto pt_chunk = Resolve(ct_program_.GetChunkIr(node.GetHandle()),
                          io_manager_.FrontendTensors());
  auto scaled_pt_chunk = ScaledPtChunk(node.GetPtLogScale(), pt_chunk);
  return parent_ct.AddCP(scaled_pt_chunk);
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const AddCS& node,
                                    const std::vector<const CtOp*>& parents) {
  const auto& ct0 = node_map_.at(parents.at(0));
  return ct0.AddCS(node.Scalar());
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const MulCC& node,
                                    const std::vector<const CtOp*>& parents) {
  (void)node;
  const auto& ct0 = node_map_.at(parents.at(0));
  const auto& ct1 = node_map_.at(parents.at(1));
  return ct0.MulCC(ct1);
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const MulCP& node,
                                    const std::vector<const CtOp*>& parents) {
  const auto& parent_ct = node_map_.at(parents.at(0));
  auto pt_chunk = Resolve(ct_program_.GetChunkIr(node.GetHandle()),
                          io_manager_.FrontendTensors());
  auto scaled_pt_chunk = ScaledPtChunk(node.GetPtLogScale(), pt_chunk);
  return parent_ct.MulCP(scaled_pt_chunk);
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const MulCS& node,
                                    const std::vector<const CtOp*>& parents) {
  const auto& ct0 = node_map_.at(parents.at(0));
  return ct0.MulCS(node.Scalar());
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const RotateC& node,
                                    const std::vector<const CtOp*>& parents) {
  const auto& parent_ct = node_map_.at(parents.at(0));
  return parent_ct.RotateC(node.RotateBy());
}

template <class CtType>
CtType Evaluator<CtType>::VisitImpl(const ZeroC& node,
                                    const std::vector<const CtOp*>& parents) {
  CHECK(parents.empty());
  return CtType::ZeroC(ct_program_.GetProgramContext(), node.GetLevelInfo());
}

}  // namespace fhelipe

#endif  // FHELIPE_EVALUATOR_H_
