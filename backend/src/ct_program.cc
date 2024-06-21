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

#include "include/ct_program.h"

#include <glog/logging.h>
#include <include/program.h>

#include <fstream>
#include <limits>
#include <memory>
#include <vector>

#include "include/add_cc.h"
#include "include/add_cp.h"
#include "include/add_cs.h"
#include "include/bootstrap_c.h"
#include "include/chunk_ir.h"
#include "include/constants.h"
#include "include/ct_op.h"
#include "include/dag.h"
#include "include/dag_io.h"
#include "include/dictionary.h"
#include "include/dictionary_impl.h"
#include "include/extended_std.h"
#include "include/filesystem_utils.h"
#include "include/input_c.h"
#include "include/io_spec.h"
#include "include/io_utils.h"
#include "include/level.h"
#include "include/level_info_utils.h"
#include "include/mul_cc.h"
#include "include/mul_cp.h"
#include "include/mul_cs.h"
#include "include/node.h"
#include "include/output_c.h"
#include "include/pass_utils.h"
#include "include/persisted_dictionary.h"
#include "include/plaintext.h"
#include "include/program_context.h"
#include "include/rescale_c.h"
#include "include/rotate_c.h"
#include "include/schedulable_mul_ksh.h"
#include "include/schedulable_rotate_ksh.h"
#include "include/t_op.h"
#include "include/utils.h"
#include "latticpp/ckks/ciphertext.h"

namespace fhelipe {
struct IoSpec;
}  // namespace fhelipe

namespace fhelipe::ct_program {

int kSecurityBits = 80;

std::vector<int> DefaultLevelToLogQMap(const Level& usable_levels,
                                       const LogScale& log_scale) {
  return Estd::transform(
      Estd::indices(usable_levels.value()),
      [&log_scale](const auto& x) { return x * log_scale.value(); });
}

namespace {

const int kCraterLakeBitsPerLevel = 28;

auto summary = std::ofstream(std::filesystem::path("/tmp/summary.txt"));

void Nikola(const std::string& s, int i) {
  summary << "Nikola " << s << " " << i << "\n";
}

int LevelToCraterLakeLevel(const Level& level, const LogScale& bits_per_level) {
  return static_cast<int>(ceil(level.value() * bits_per_level.value() /
                               static_cast<double>(kCraterLakeBitsPerLevel)));
}

int LevelToAxelSlots(const Level& level,
                     const std::vector<int>& level_to_craterlake_level_map) {
  auto result = 2 * level_to_craterlake_level_map.at(level.value());
  return result;
}

int LevelToR(const Level& level,
             const std::vector<int>& level_to_craterlake_level_map) {
  return level_to_craterlake_level_map.at(level.value());
}

template <typename T>
void WriteSchedulableNode(std::ostream& stream, const T& node,
                          const std::vector<int>& level_to_craterlake_level_map,
                          const std::vector<int>& level_to_log_q_map) = delete;

template <>
void WriteSchedulableNode<InputC>(
    std::ostream& stream, const InputC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "CIPHERTEXT");
  WriteStream<std::string>(stream, "\t");
  WriteStream<std::string>(stream, ToFilename(node.GetIoSpec()));
  WriteStream<std::string>(stream, "\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
}

template <>
void WriteSchedulableNode<ZeroC>(
    std::ostream& stream, const ZeroC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "CIPHERTEXT");
  WriteStream<std::string>(stream, "\t");
  WriteStream<std::string>(stream, "ZERO 0");
  WriteStream<std::string>(stream, "\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
}

int GetLogQ(const Level& level, const std::vector<int>& level_to_log_q_map) {
  return level_to_log_q_map[level.value() - 1];
}

int GetKshDigits(int log_q) {
  // TODO
  if (kSecurityBits == 80) {
    if (log_q > 60 * 28) {
      LOG(FATAL) << "Not allowed for 80-bit security!";
    } else if (log_q > 52 * 28) {
      return 2;
    } else {
      return 1;
    }
  } else if (kSecurityBits == 128) {
    if (log_q > 51 * 128) {
      LOG(FATAL) << "Not allowed for 128-bit security!";
    } else if (log_q > 43 * 28) {
      return 3;
    } else if (log_q > 32 * 28) {
      return 2;
    } else {
      return 1;
    }
  }
  LOG(FATAL);
}

std::string MulCCType(const Level& level,
                      const std::vector<int>& level_to_log_q_map) {
  auto log_q = GetLogQ(level, level_to_log_q_map);
  auto ksh_digits = GetKshDigits(log_q);
  if (ksh_digits == 1) {
    return "MUL_KS_NEW";
  } else if (ksh_digits == 2) {
    return "MUL_KS_2DIGIT";
  } else if (ksh_digits == 3) {
    return "MUL_KS_3DIGIT";
  }
  LOG(FATAL) << "Not supported KSH digits: " << ksh_digits;
}

std::string RotateCType(const Level& level,
                        const std::vector<int>& level_to_log_q_map) {
  auto log_q = GetLogQ(level, level_to_log_q_map);
  auto ksh_digits = GetKshDigits(log_q);
  if (ksh_digits == 1) {
    return "ROTATE_KS_NEW";
  } else if (ksh_digits == 2) {
    return "ROTATE_KS_2DIGIT";
  } else if (ksh_digits == 3) {
    return "ROTATE_KS_3DIGIT";
  }
  LOG(FATAL) << "Not supported KSH digits: " << ksh_digits;
}

template <>
void WriteSchedulableNode<MulCC>(
    std::ostream& stream, const MulCC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(
      stream, MulCCType(node.GetLevel(), level_to_log_q_map) + "\tmul\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  Nikola("MulCC", LevelToR(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
}

template <>
void WriteSchedulableNode<MulCP>(
    std::ostream& stream, const MulCP& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "MUL_SIMPLE\tMulCP\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  Nikola("MulCP", LevelToR(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
}

template <>
void WriteSchedulableNode<AddCC>(
    std::ostream& stream, const AddCC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "ADD\tadd\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
  Nikola("AddCC", LevelToR(node.GetLevel(), level_to_craterlake_level_map));
}

template <>
void WriteSchedulableNode<AddCP>(
    std::ostream& stream, const AddCP& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "ADD\tadd\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
  Nikola("AddCP", LevelToR(node.GetLevel(), level_to_craterlake_level_map));
}

template <>
void WriteSchedulableNode<RotateC>(
    std::ostream& stream, const RotateC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(
      stream, RotateCType(node.GetLevel(), level_to_log_q_map) + "\trotate\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
  Nikola("RotateC", LevelToR(node.GetLevel(), level_to_craterlake_level_map));
}

template <>
void WriteSchedulableNode<BootstrapC>(
    std::ostream& stream, const BootstrapC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  // Treat a bootstrap as a new input for simplicity
  WriteStream<std::string>(stream, "CIPHERTEXT");
  WriteStream<std::string>(stream, "\tBOOTSTRAPPED\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
  Nikola("BootstrapC", 0);
}

template <>
void WriteSchedulableNode<RescaleC>(
    std::ostream& stream, const RescaleC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "RESCALE");
  WriteStream<std::string>(stream, "\trescale\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
}

template <>
void WriteSchedulableNode<OutputC>(
    std::ostream& stream, const OutputC& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  // There is no concept of output in the scheduler, so we just create a fake
  // MulCP node. This has minimal performance impact and makes this code
  // cleaner.
  WriteStream<std::string>(stream, "MUL_SIMPLE\tMulCP\t");
  WriteStream<int>(
      stream, LevelToAxelSlots(node.GetLevel(), level_to_craterlake_level_map));
  WriteStream<std::string>(stream, "\tct");
}

template <>
void WriteSchedulableNode<SchedulableMulKsh>(
    std::ostream& stream, const SchedulableMulKsh& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "KSH\t");
  stream << "ksh(" << node.GetLevel().value() << ", "
         << "mul)\t"
         << (GetKshDigits(GetLogQ(node.GetLevel(), level_to_log_q_map)) + 1) *
                level_to_craterlake_level_map.at(node.GetLevel().value())
         << "\tksh";
}

template <>
void WriteSchedulableNode<SchedulableRotateKsh>(
    std::ostream& stream, const SchedulableRotateKsh& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteStream<std::string>(stream, "KSH\t");
  stream << "ksh(" << node.GetLevel().value() << ", " << node.RotateBy()
         << ")\t"
         << (GetKshDigits(GetLogQ(node.GetLevel(), level_to_log_q_map)) + 1) *
                level_to_craterlake_level_map.at(node.GetLevel().value())
         << "\tksh";
}

template <>
void WriteSchedulableNode<CtOp>(
    std::ostream& stream, const CtOp& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  if (const auto* zero_c = dynamic_cast<const ZeroC*>(&node)) {
    WriteSchedulableNode(stream, *zero_c, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* input_c = dynamic_cast<const InputC*>(&node)) {
    WriteSchedulableNode(stream, *input_c, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* mul_cc = dynamic_cast<const MulCC*>(&node)) {
    WriteSchedulableNode(stream, *mul_cc, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* mul_cp = dynamic_cast<const MulCP*>(&node)) {
    WriteSchedulableNode(stream, *mul_cp, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* add_cc = dynamic_cast<const AddCC*>(&node)) {
    WriteSchedulableNode(stream, *add_cc, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* add_cp = dynamic_cast<const AddCP*>(&node)) {
    WriteSchedulableNode(stream, *add_cp, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* rotate_c = dynamic_cast<const RotateC*>(&node)) {
    WriteSchedulableNode(stream, *rotate_c, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* bootstrap_c = dynamic_cast<const BootstrapC*>(&node)) {
    WriteSchedulableNode(stream, *bootstrap_c, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* rescale_c = dynamic_cast<const RescaleC*>(&node)) {
    // TODO: Add destination slots
    Nikola("RescaleC",
           LevelToR(node.GetLevel(), level_to_craterlake_level_map));
    WriteSchedulableNode(stream, *rescale_c, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* output_c = dynamic_cast<const OutputC*>(&node)) {
    WriteSchedulableNode(stream, *output_c, level_to_craterlake_level_map,
                         level_to_log_q_map);
  } else if (const auto* schedulable_mul_ksh =
                 dynamic_cast<const SchedulableMulKsh*>(&node)) {
    WriteSchedulableNode(stream, *schedulable_mul_ksh,
                         level_to_craterlake_level_map, level_to_log_q_map);
  } else if (const auto* schedulable_rotate_ksh =
                 dynamic_cast<const SchedulableRotateKsh*>(&node)) {
    WriteSchedulableNode(stream, *schedulable_rotate_ksh,
                         level_to_craterlake_level_map, level_to_log_q_map);
  } else {
    LOG(FATAL) << "Unrecognize CtOp";
  }
}

template <>
void WriteSchedulableNode<Node<CtOp>>(
    std::ostream& stream, const Node<CtOp>& node,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  WriteSchedulableNode(stream, node.Value(), level_to_craterlake_level_map,
                       level_to_log_q_map);
  WriteStream<std::string>(stream, "\n");
}

int kScratchpadMegabytes = 256;

class KshDictionary {
 public:
  std::shared_ptr<Node<CtOp>> At(const Level& level) {
    if (!Estd::contains_key(mul_kshs, level.value())) {
      mul_kshs.emplace(level.value(),
                       std::make_shared<Node<CtOp>>(
                           std::make_unique<SchedulableMulKsh>(level)));
    }
    return mul_kshs.at(level.value());
  }

  std::shared_ptr<Node<CtOp>> At(const Level& level, int rotate_by) {
    if (!Estd::contains_key(rotate_kshs[level.value()], rotate_by)) {
      rotate_kshs[level.value()].emplace(
          rotate_by,
          std::make_shared<Node<CtOp>>(std::make_unique<SchedulableRotateKsh>(
              level.value(), rotate_by)));
    }
    return rotate_kshs.at(level.value()).at(rotate_by);
  }

  std::shared_ptr<Node<CtOp>> At(const CtOp& ct_op) {
    if (const auto* mul_cc = dynamic_cast<const MulCC*>(&ct_op)) {
      return At(mul_cc->GetLevel());
    }
    if (const auto* rotate_c = dynamic_cast<const RotateC*>(&ct_op)) {
      return At(rotate_c->GetLevel(), rotate_c->RotateBy());
    }
    LOG(FATAL);
  }

 private:
  std::unordered_map<int, std::shared_ptr<Node<CtOp>>> mul_kshs;
  std::unordered_map<int, std::unordered_map<int, std::shared_ptr<Node<CtOp>>>>
      rotate_kshs;
};

bool RequiresKeyswitching(const CtOp& node) {
  return dynamic_cast<const MulCC*>(&node) ||
         dynamic_cast<const RotateC*>(&node);
}

Dag<CtOp> AddSchedulableKshNodes(const ct_program::CtProgram& ct_program) {
  auto result = CloneFromAncestor(ct_program.GetDag());
  KshDictionary kshs;
  for (const auto& node : result.NodesInAncestorIdOrder()) {
    if (RequiresKeyswitching(node->Value())) {
      auto ksh = kshs.At(node->Value());
      if (ksh->Parents().empty()) {
        AddParentChildEdge(node->Parents()[0], ksh);
      }
      AddParentChildEdge(ksh, node);
    }
  }
  return result;
}

void WriteSchedulableParentChildEdge(std::ostream& stream,
                                     const std::unordered_map<int, int>& id_map,
                                     const Node<CtOp>& parent,
                                     const Node<CtOp>& child) {
  WriteStream<int>(stream, id_map.at(parent.NodeId()));
  stream << "\t";
  WriteStream<int>(stream, id_map.at(child.NodeId()));
  stream << "\n";
}

}  // namespace

CtProgram::CtProgram(const ProgramContext& ct_param,
                     std::unique_ptr<Dictionary<ChunkIr>>&& chunk_dict,
                     Dag<CtOp>&& dag)
    : ct_param_(ct_param),
      ct_op_dag_(std::move(dag)),
      chunk_dict_(std::move(chunk_dict)) {}

std::shared_ptr<Node<CtOp>> CreateBootstrapC(
    CtProgram& ct_program, const LevelInfo& level_info,
    const std::shared_ptr<Node<CtOp>>& parent) {
  return ct_program.AddNode(std::make_unique<BootstrapC>(level_info), {parent});
}

std::shared_ptr<Node<CtOp>> CreateRescaleC(
    CtProgram& ct_program, const LevelInfo& level_info,
    const std::shared_ptr<Node<CtOp>>& parent) {
  return ct_program.AddNode(std::make_unique<RescaleC>(level_info), {parent});
}

std::shared_ptr<Node<CtOp>> CreateInputC(CtProgram& ct_program,
                                         const LevelInfo& level_info,
                                         const IoSpec& io_spec) {
  return ct_program.AddNode(std::make_unique<InputC>(level_info, io_spec), {});
}

std::shared_ptr<Node<CtOp>> FetchZeroC(const std::shared_ptr<Node<CtOp>>& node,
                                       const LevelInfo& level_info) {
  auto sentinel = node->Sentinel();
  auto candidates =
      Estd::filter(sentinel->Children(), [&level_info](const auto& ptr) {
        const auto* zero_c = dynamic_cast<const ZeroC*>(&ptr->Value());
        return zero_c && zero_c->GetLevelInfo() == level_info;
      });

  if (!candidates.empty()) {
    return *candidates.begin();
  }

  auto zero_c =
      std::make_shared<Node<CtOp>>(std::make_unique<ZeroC>(level_info));
  AddParentChildEdge(sentinel, zero_c);
  return zero_c;
}

std::vector<int> BestPossibleLevelToCraterLakeLevelMap(
    const Level& max_levels, const LogScale& bits_per_level) {
  std::vector<int> result{0};
  for (int level : Estd::indices(1, 1 + max_levels.value())) {
    result.push_back(LevelToCraterLakeLevel(Level(level), bits_per_level));
  }
  return result;
}

std::shared_ptr<Node<CtOp>> CreateOutputC(
    CtProgram& ct_program, const LevelInfo& level_info, const IoSpec& io_spec,
    const std::shared_ptr<Node<CtOp>>& parent) {
  return ct_program.AddNode(std::make_unique<OutputC>(level_info, io_spec),
                            {parent});
}

std::shared_ptr<Node<CtOp>> CreateAddCC(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent_1,
    const std::shared_ptr<Node<CtOp>>& parent_2) {
  if (dynamic_cast<const ZeroC*>(&parent_2->Value())) {
    return parent_1;
  }
  if (dynamic_cast<const ZeroC*>(&parent_1->Value())) {
    return parent_2;
  }
  return ct_program.AddNode(
      std::make_unique<AddCC>(
          LevelInfo(std::min(parent_1->Value().GetLevelInfo().Level(),
                             parent_2->Value().GetLevelInfo().Level()),
                    std::max(parent_1->Value().GetLevelInfo().LogScale(),
                             parent_2->Value().GetLevelInfo().LogScale()))),
      {parent_1, parent_2});
}

std::shared_ptr<Node<CtOp>> CreateAddCP(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ChunkIr& chunk, LogScale pt_log_scale) {
  auto handle = ct_program.RecordChunk(chunk);
  return ct_program.AddNode(
      std::make_unique<AddCP>(
          LevelInfo{parent->Value().GetLevelInfo().Level(),
                    LogScale(std::max(
                        parent->Value().GetLevelInfo().LogScale().value(),
                        pt_log_scale.value()))},
          handle, pt_log_scale),
      {parent});
}

std::shared_ptr<Node<CtOp>> CreateAddCS(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ScaledPtVal& scalar) {
  return ct_program.AddNode(
      std::make_unique<AddCS>(
          LevelInfo(parent->Value().GetLevelInfo().Level(),
                    std::max(parent->Value().LogScale(), scalar.GetLogScale())),
          scalar),
      {parent});
}

std::shared_ptr<Node<CtOp>> CreateMulCC(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent_1,
    const std::shared_ptr<Node<CtOp>>& parent_2) {
  auto level_info =
      LevelInfo{std::min(parent_1->Value().GetLevelInfo().Level(),
                         parent_2->Value().GetLevelInfo().Level()),
                parent_1->Value().GetLevelInfo().LogScale() +
                    parent_2->Value().GetLevelInfo().LogScale()};
  if (dynamic_cast<const ZeroC*>(&parent_2->Value()) ||
      dynamic_cast<const ZeroC*>(&parent_1->Value())) {
    return ct_program::FetchZeroC(parent_1, level_info);
  }
  return ct_program.AddNode(std::make_unique<MulCC>(level_info),
                            {parent_1, parent_2});
}

std::shared_ptr<Node<CtOp>> CreateMulCP(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ChunkIr& chunk, LogScale pt_log_scale) {
  auto handle = ct_program.RecordChunk(chunk);
  if (dynamic_cast<const ZeroC*>(&parent->Value())) {
    return FetchZeroCThatIsAtSameLevelInfoAsAMulCPChildOf(parent, pt_log_scale);
  }
  return ct_program.AddNode(
      std::make_unique<MulCP>(
          LevelInfo(parent->Value().GetLevelInfo().Level(),
                    pt_log_scale + parent->Value().GetLevelInfo().LogScale()),
          handle, pt_log_scale),
      {parent});
}

std::shared_ptr<Node<CtOp>> CreateMulCS(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    const ScaledPtVal& scalar) {
  if (dynamic_cast<const ZeroC*>(&parent->Value())) {
    return FetchZeroCThatIsAtSameLevelInfoAsAMulCPChildOf(parent,
                                                          scalar.GetLogScale());
  }
  return ct_program.AddNode(
      std::make_unique<MulCS>(
          LevelInfo(
              parent->Value().GetLevelInfo().Level(),
              scalar.GetLogScale() + parent->Value().GetLevelInfo().LogScale()),
          scalar),
      {parent});
}

std::shared_ptr<Node<CtOp>> CreateRotateC(
    CtProgram& ct_program, const std::shared_ptr<Node<CtOp>>& parent,
    int rotate_by) {
  if (dynamic_cast<const ZeroC*>(&parent->Value())) {
    return parent;
  }
  if (rotate_by == 0 ||
      rotate_by ==
          (1 << ct_program.GetProgramContext().GetLogChunkSize().value())) {
    return parent;
  }
  return ct_program.AddNode(
      std::make_unique<RotateC>(parent->Value().GetLevelInfo(), rotate_by),
      {parent});
}

// TODO(nsamar): Change Axel's scheduler to work with fhelipe's ct_programs
void WriteSchedulableDataflowGraph(
    std::ostream& stream, const ct_program::CtProgram& ct_program,
    const std::vector<int>& level_to_craterlake_level_map,
    const std::vector<int>& level_to_log_q_map) {
  auto schedulable_dag = AddSchedulableKshNodes(ct_program);
  WriteStream<int>(stream, kScratchpadMegabytes);
  stream << "\n";

  int count = 0;
  std::unordered_map<int, int> node_id_to_axel_id;
  for (const auto& node_t : schedulable_dag.NodesInAncestorIdOrder()) {
    WriteStream<int>(stream, count);
    WriteStream<std::string>(stream, "\t");
    node_id_to_axel_id[node_t->NodeId()] = count++;
    WriteSchedulableNode(stream, *node_t, level_to_craterlake_level_map,
                         level_to_log_q_map);
    // If it is MulCC, then there will be no moddown, cuz I need to rescale
    // first Same for MulCP, BootstrapC
    if (!dynamic_cast<const MulCC*>(&node_t->Value()) &&
        !dynamic_cast<const MulCP*>(&node_t->Value()) &&
        !dynamic_cast<const BootstrapC*>(&node_t->Value())) {
      auto curr_lvl = node_t->Value().GetLevel();
      auto num_diff_child_levels =
          Estd::filter(Estd::transform(
                           node_t->Children(),
                           [](const auto& x) { return x->Value().GetLevel(); }),
                       [&curr_lvl](const auto& lvl) { return lvl != curr_lvl; })
              .size();
      for (int i : Estd::indices(num_diff_child_levels)) {
        (void)i;
        Nikola("ModDownC", LevelToR(node_t->Value().GetLevel().value(),
                                    level_to_craterlake_level_map));
      }
    }
  }

  for (const auto& node_t : schedulable_dag.NodesInAncestorIdOrder()) {
    for (const auto& child_t : node_t->Children()) {
      if (!dynamic_cast<const SchedulableRotateKsh*>(&child_t->Value()) &&
          !dynamic_cast<const SchedulableMulKsh*>(&child_t->Value())) {
        WriteSchedulableParentChildEdge(stream, node_id_to_axel_id, *node_t,
                                        *child_t);
      }
    }
  }
}

TOp::Chunk FetchZeroCAtSameLevelInfoAs(const TOp::Chunk& node) {
  return FetchZeroC(node, node->Value().GetLevelInfo());
}

TOp::Chunk FetchZeroCThatIsAtSameLevelInfoAsAMulCPChildOf(
    const TOp::Chunk& parent, LogScale pt_log_scale) {
  auto level_info_after_mul_cp =
      LevelInfo(parent->Value().GetLevelInfo().Level(),
                pt_log_scale + parent->Value().GetLevelInfo().LogScale());
  return ct_program::FetchZeroC(parent, level_info_after_mul_cp);
}

}  // namespace fhelipe::ct_program

namespace fhelipe {

std::vector<ct_program::CtProgram> PartitionProgram(
    const ct_program::CtProgram& program) {
  std::vector<ct_program::CtProgram> result;
  std::unordered_map<std::shared_ptr<Node<CtOp>>, int> node_to_partition;
  std::unordered_map<std::shared_ptr<Node<CtOp>>, std::shared_ptr<Node<CtOp>>>
      old_to_new_nodes;

  for (const auto& node : program.NodesInTopologicalOrder()) {
    int partition = 0;
    if (!node->Parents().empty()) {
      partition = Estd::max_element(Estd::transform(
          node->Parents(), [&node_to_partition](const auto& parent) -> int {
            return node_to_partition.at(parent);
          }));
    }
    if (dynamic_cast<const BootstrapC*>(&node->Value())) {
      partition += 1;
    }
    if (result.size() <= partition) {
      result.emplace_back(program.GetProgramContext(),
                          std::make_unique<RamDictionary<ChunkIr>>(),
                          Dag<CtOp>());
    }
    std::vector<std::shared_ptr<Node<CtOp>>> parents = Estd::transform(
        Estd::filter(node->Parents(),
                     [partition, &node_to_partition](const auto& parent) {
                       return (partition == node_to_partition.at(parent));
                     }),
        [&old_to_new_nodes](
            const auto& old_parent) -> std::shared_ptr<Node<CtOp>> {
          return old_to_new_nodes.at(old_parent);
        });

    node_to_partition.emplace(node, partition);
    if (!parents.empty()) {
      old_to_new_nodes.emplace(node, result.at(partition).AddNode(
                                         node->Value().CloneUniq(), parents));
    } else {
      // Create fake input node
      old_to_new_nodes.emplace(
          node, result.at(partition).AddNode(
                    std::make_unique<InputC>(
                        node->Value().GetLevelInfo(),
                        IoSpec("phony_" + std::to_string(node->NodeId()), 15)),
                    parents));
    }
  }

  return result;
}

template <>
void WriteStream<ct_program::CtProgram>(
    std::ostream& stream, const ct_program::CtProgram& ct_program) {
  WriteStream<ProgramContext>(stream, ct_program.GetProgramContext());
  stream << '\n';
  WriteStream<Dictionary<ChunkIr>>(stream, *ct_program.ChunkDictionary());
  stream << '\n';
  WriteStream<Dag<CtOp>>(stream, ct_program.GetDag());
}

template <>
ct_program::CtProgram ReadStream<ct_program::CtProgram>(std::istream& stream) {
  auto context = ReadStream<ProgramContext>(stream);
  std::unique_ptr<Dictionary<ChunkIr>> chunk_dict =
      Dictionary<ChunkIr>::CreateInstance(stream);
  auto ct_op_dag = ReadStream<Dag<CtOp>>(stream);
  return {context, std::move(chunk_dict), std::move(ct_op_dag)};
}

std::vector<std::shared_ptr<Node<CtOp>>>
ct_program::CtProgram::NodesInTopologicalOrder() const {
  return ct_op_dag_.NodesInTopologicalOrder();
}

}  // namespace fhelipe
