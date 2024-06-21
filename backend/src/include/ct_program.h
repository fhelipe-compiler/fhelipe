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

#ifndef FHELIPE_CT_PROGRAM_H_
#define FHELIPE_CT_PROGRAM_H_

#include <glog/logging.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chunk_ir.h"
#include "ct_op.h"
#include "dag.h"
#include "dag_io.h"
#include "io_c.h"
#include "io_spec.h"
#include "persisted_dictionary.h"
#include "plaintext.h"
#include "program_context.h"
#include "scaled_pt_val.h"

namespace fhelipe::ct_program {

extern int kSecurityBits;

std::vector<int> BestPossibleLevelToCraterLakeLevelMap(
    const Level& max_levels, const LogScale& bits_per_level);

std::vector<int> DefaultLevelToLogQMap(const Level& usable_levels,
                                       const LogScale& log_scale);

TOp::Chunk FetchZeroCAtSameLevelInfoAs(const TOp::Chunk& node);

TOp::Chunk FetchZeroCThatIsAtSameLevelInfoAsAMulCPChildOf(
    const TOp::Chunk& parent, LogScale pt_log_scale);

class CtProgram {
 public:
  CtProgram(const ProgramContext& ct_param,
            const Dictionary<ChunkIr>& chunk_dict);
  CtProgram(const ProgramContext& ct_param,
            std::unique_ptr<Dictionary<ChunkIr>>&& chunk_dict, Dag<CtOp>&& dag);

  CtProgram(CtProgram&&) = default;
  virtual ~CtProgram() = default;
  CtProgram(const CtProgram&) = delete;
  CtProgram& operator=(const CtProgram&) = delete;
  CtProgram& operator=(CtProgram&&) = default;

  std::shared_ptr<Node<CtOp>> GetNodeById(int node_id) const {
    return ct_op_dag_.GetNodeById(node_id);
  }
  std::shared_ptr<Node<CtOp>> AddNode(
      std::unique_ptr<CtOp>&& new_node,
      const std::vector<std::shared_ptr<Node<CtOp>>>& parents);
  std::shared_ptr<Node<CtOp>> AddNode(
      int node_id, std::unique_ptr<CtOp>&& new_node,
      const std::vector<std::shared_ptr<Node<CtOp>>>& parents);
  KeyType RecordChunk(const ChunkIr& chunk);
  ChunkIr GetChunkIr(const KeyType& key) const { return chunk_dict_->At(key); }

  Dictionary<ChunkIr>* ChunkDictionary() const { return chunk_dict_.get(); }

  std::vector<std::shared_ptr<Node<CtOp>>> NodesInTopologicalOrder() const;
  const ProgramContext& GetProgramContext() const { return ct_param_; }
  const Dag<CtOp>& GetDag() const { return ct_op_dag_; }

  Dag<CtOp>& GetDag() { return ct_op_dag_; }

 private:
  ProgramContext ct_param_;
  Dag<CtOp> ct_op_dag_;
  std::set<IoSpec> io_specs_;
  std::unique_ptr<Dictionary<ChunkIr>> chunk_dict_;

  void RegisterNewIoNode(const IoC* ioc);
  void RegisterAddedNode(const CtOp& new_node);
};

inline void CtProgram::RegisterNewIoNode(const IoC* ioc) {
  CHECK(ioc);
  CHECK(!io_specs_.contains(ioc->GetIoSpec()));
  io_specs_.insert(ioc->GetIoSpec());
}

std::shared_ptr<Node<CtOp>> CreateInputC(CtProgram& ct_program,
                                         const LevelInfo& level_info,
                                         const IoSpec& io_spec);

std::shared_ptr<Node<CtOp>> FetchZeroC(const std::shared_ptr<Node<CtOp>>& node,
                                       const LevelInfo& level_info);

std::shared_ptr<Node<CtOp>> CreateOutputC(
    CtProgram& ct_program, const LevelInfo& level_info, const IoSpec& io_spec,
    const std::shared_ptr<Node<CtOp>>& parent);

std::shared_ptr<Node<CtOp>> CreateMulCC(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent_1,
    const std::shared_ptr<Node<CtOp>>& parent_2);
std::shared_ptr<Node<CtOp>> CreateMulCP(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ChunkIr& chunk, LogScale pt_log_scale);
std::shared_ptr<Node<CtOp>> CreateMulCS(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ScaledPtVal& scalar);

std::shared_ptr<Node<CtOp>> CreateAddCC(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent_1,
    const std::shared_ptr<Node<CtOp>>& parent_2);
std::shared_ptr<Node<CtOp>> CreateAddCP(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ChunkIr& chunk, LogScale log_scale);
std::shared_ptr<Node<CtOp>> CreateAddCS(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ScaledPtVal& scalar);
std::shared_ptr<Node<CtOp>> CreateRotateC(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    int rotate_by);
std::shared_ptr<Node<CtOp>> CreateBootstrapC(
    CtProgram& ct_program, const LevelInfo& level_info,
    const std::shared_ptr<Node<CtOp>>& parent);

std::shared_ptr<Node<CtOp>> CreateRescaleC(
    CtProgram& ct_program, const LevelInfo& level_info,
    const std::shared_ptr<Node<CtOp>>& parent);

inline CtProgram::CtProgram(const ProgramContext& ct_param,
                            const Dictionary<ChunkIr>& chunk_dict)
    : CtProgram(ct_param, chunk_dict.CloneUniq(), Dag<CtOp>()) {}

inline void CtProgram::RegisterAddedNode(const CtOp& new_node) {
  if (const auto* io_node = dynamic_cast<const IoC*>(&new_node)) {
    RegisterNewIoNode(io_node);
  }
}

inline std::shared_ptr<Node<CtOp>> CtProgram::AddNode(
    int node_id, std::unique_ptr<CtOp>&& new_node,
    const std::vector<std::shared_ptr<Node<CtOp>>>& parents) {
  RegisterAddedNode(*new_node);
  return ct_op_dag_.AddNode(node_id, std::move(new_node), parents);
}

inline std::shared_ptr<Node<CtOp>> CtProgram::AddNode(
    std::unique_ptr<CtOp>&& new_node,
    const std::vector<std::shared_ptr<Node<CtOp>>>& parents) {
  RegisterAddedNode(*new_node);
  return ct_op_dag_.AddNode(std::move(new_node), parents);
}

inline KeyType CtProgram::RecordChunk(const ChunkIr& chunk) {
  return chunk_dict_->Record(chunk);
}

void WriteSchedulableDataflowGraph(
    std::ostream& stream, const ct_program::CtProgram& ct_program,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map);

}  // namespace fhelipe::ct_program

namespace fhelipe {
template <>
void WriteStream<ct_program::CtProgram>(
    std::ostream& stream, const ct_program::CtProgram& ct_program);

template <>
ct_program::CtProgram ReadStream<ct_program::CtProgram>(std::istream& stream);

std::vector<ct_program::CtProgram> PartitionProgram(
    const ct_program::CtProgram& program);

}  // namespace fhelipe

#endif  // FHELIPE_CT_PROGRAM_H_
