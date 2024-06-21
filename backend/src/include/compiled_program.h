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

#ifndef FHELIPE_COMPILED_PROGRAM_H_
#define FHELIPE_COMPILED_PROGRAM_H_

#include <iostream>

#include "debug_info.h"
#include "include/debug_info_archive.h"
#include "io_utils.h"
#include "latticpp/ckks/ciphertext.h"
#include "pass_utils.h"
#include "program.h"

namespace fhelipe {

class Compiler;

std::string PaddedIntString(int value);

class PassId {
 public:
  PassId(int pass_idx, const PassName& pass_name)
      : pass_idx_(pass_idx), pass_name_(pass_name) {}
  int PassIndex() const { return pass_idx_; }
  const PassName& GetPassName() const { return pass_name_; }

 private:
  int pass_idx_;
  PassName pass_name_;
};

template <>
void WriteStream<PassId>(std::ostream& stream, const PassId& pass_id);

template <>
PassId ReadStream<PassId>(std::istream& stream);

class CompiledProgram {
 public:
  CompiledProgram(const Program& program,
                  std::pair<PassId, ParserOutput>&& input_dag,
                  std::vector<std::pair<PassId, LayoutOptimizerOutput>>&&
                      layout_optimizer_dags,
                  std::pair<PassId, RescalingPassOutput>&& rescaling_pass_dag,
                  std::vector<std::pair<PassId, LevelingPassOutput>>&&
                      leveling_optimizer_dags,
                  std::vector<std::pair<PassId, CtOpOptimizerOutput>>&&
                      ct_op_optimizer_dags);
  std::string FrontendCode() const { return program_.Code(); }

  const PassName& LastPassName() const {
    return ct_op_optimizer_dags_.back().first.GetPassName();
  }

  const std::filesystem::path& ExeFolder() const {
    return program_.ExeFolder();
  }
  const CtOpOptimizerOutput& Result() const;
  std::vector<PassId> GetPassIds() const;

  template <typename T>
  const T& GetDag(const PassName& pass_name) const;

  friend void WriteStreamFriend(std::ostream& stream,
                                const CompiledProgram& cp);

  template <typename SrcDagT, typename DestDagT, typename SrcT, typename DestT>
  DebugInfo<SrcDagT, DestDagT, SrcT, DestT> GetDebugInfo(
      const PassName& source_pass, const PassName& destination_pass) const;

  PassName LastLevelingPassName() const {
    return leveling_optimizer_dags_.back().first.GetPassName();
  }

 private:
  Program program_;
  std::pair<PassId, ParserOutput> input_dag_;
  std::vector<std::pair<PassId, LayoutOptimizerOutput>> layout_optimizer_dags_;
  std::pair<PassId, RescalingPassOutput> rescaling_pass_dag_;
  std::vector<std::pair<PassId, LevelingOptimizerOutput>>
      leveling_optimizer_dags_;
  std::vector<std::pair<PassId, CtOpOptimizerOutput>> ct_op_optimizer_dags_;

  std::vector<std::pair<PassId, DebugInfoArchive>> archives_;

  std::vector<DebugInfoArchive> GetDebugInfoArchivesBetween(
      const PassName& source_pass, const PassName& destination_pass) const;
  int FindPassIndex(const PassName& pass_name) const;
  DebugInfoArchive GetDebugInfoArchive(const PassName& source_pass,
                                       const PassName& destination_pass) const;
};

template <typename SrcDagT, typename DestDagT, typename SrcT, typename DestT>
DebugInfo<SrcDagT, DestDagT, SrcT, DestT> CompiledProgram::GetDebugInfo(
    const PassName& source_pass, const PassName& destination_pass) const {
  return {&GetDag<SrcDagT>(source_pass), &GetDag<DestDagT>(destination_pass),
          ClusterDebugInfoArchive(
              GetDebugInfoArchive(source_pass, destination_pass))};
}

template <>
inline const ParserOutput& CompiledProgram::GetDag<ParserOutput>(
    const PassName& pass_name) const {
  CHECK(pass_name == input_dag_.first.GetPassName());
  return input_dag_.second;
}

template <>
inline const LayoutOptimizerOutput&
CompiledProgram::GetDag<LayoutOptimizerOutput>(
    const PassName& pass_name) const {
  for (const auto& optimizer : layout_optimizer_dags_) {
    if (optimizer.first.GetPassName() == pass_name) {
      return optimizer.second;
    }
  }
  LOG(FATAL);
}

template <>
inline const RescalingPassOutput& CompiledProgram::GetDag<RescalingPassOutput>(
    const PassName& pass_name) const {
  CHECK(pass_name == rescaling_pass_dag_.first.GetPassName());
  return rescaling_pass_dag_.second;
}

template <>
inline const LevelingOptimizerOutput&
CompiledProgram::GetDag<LevelingOptimizerOutput>(
    const PassName& pass_name) const {
  for (const auto& optimizer : leveling_optimizer_dags_) {
    if (optimizer.first.GetPassName() == pass_name) {
      return optimizer.second;
    }
  }
  LOG(FATAL);
}

template <>
inline const CtOpOptimizerOutput& CompiledProgram::GetDag<CtOpOptimizerOutput>(
    const PassName& pass_name) const {
  for (const auto& optimizer : ct_op_optimizer_dags_) {
    if (optimizer.first.GetPassName() == pass_name) {
      return optimizer.second;
    }
  }
  LOG(FATAL);
}

template <>
inline void WriteStream<CompiledProgram>(std::ostream& stream,
                                         const CompiledProgram& cp) {
  WriteStreamFriend(stream, cp);
}

template <>
CompiledProgram ReadStream<CompiledProgram>(std::istream& stream);

// Convenience class for accumulating all the stuff you need to construct a
// CompiledProgram
class CompiledProgramBuilder {
 public:
  explicit CompiledProgramBuilder(const Compiler* compiler,
                                  const Program& program)
      : compiler_(compiler), program_(program) {}

  template <typename OutputT>
  OutputT* RecordPass(const PassName& pass_ptr, OutputT&& output);

  CompiledProgram GetCompiledProgram();

 private:
  const Compiler* compiler_;
  int pass_count_ = 0;

  Program program_;
  std::optional<std::pair<PassId, ParserOutput>> input_dag_ = std::nullopt;
  std::vector<std::pair<PassId, LayoutPassOutput>> layout_optimizer_dags_;

  std::optional<std::pair<PassId, RescalingPassOutput>> rescaling_pass_dag_ =
      std::nullopt;

  std::vector<std::pair<PassId, LevelingOptimizerOutput>>
      leveling_optimizer_dags_;

  std::vector<std::pair<PassId, CtOpOptimizerOutput>> ct_op_optimizer_dags_;
};

template <>
inline ParserOutput* CompiledProgramBuilder::RecordPass<ParserOutput>(
    const PassName& pass_name, ParserOutput&& output) {
  input_dag_ =
      std::make_pair(PassId(pass_count_++, pass_name), std::move(output));
  return &input_dag_.value().second;
}

template <>
inline LayoutOptimizerOutput*
CompiledProgramBuilder::RecordPass<LayoutOptimizerOutput>(
    const PassName& pass_name, LayoutOptimizerOutput&& output) {
  layout_optimizer_dags_.emplace_back(
      std::make_pair(PassId(pass_count_++, pass_name), std::move(output)));
  return &layout_optimizer_dags_.back().second;
}

template <>
inline RescalingPassOutput*
CompiledProgramBuilder::RecordPass<RescalingPassOutput>(
    const PassName& pass_name, RescalingPassOutput&& output) {
  rescaling_pass_dag_ =
      std::make_pair(PassId(pass_count_++, pass_name), std::move(output));
  return &rescaling_pass_dag_.value().second;
}

template <>
inline LevelingOptimizerOutput*
CompiledProgramBuilder::RecordPass<LevelingOptimizerOutput>(
    const PassName& pass_name, LevelingOptimizerOutput&& output) {
  leveling_optimizer_dags_.emplace_back(
      std::make_pair(PassId(pass_count_++, pass_name), std::move(output)));
  return &leveling_optimizer_dags_.back().second;
}

template <>
inline CtOpOptimizerOutput*
CompiledProgramBuilder::RecordPass<CtOpOptimizerOutput>(
    const PassName& pass_name, CtOpOptimizerOutput&& output) {
  ct_op_optimizer_dags_.emplace_back(
      std::make_pair(PassId(pass_count_++, pass_name), std::move(output)));
  return &ct_op_optimizer_dags_.back().second;
}

inline CompiledProgram CompiledProgramBuilder::GetCompiledProgram() {
  return {program_,
          std::move(input_dag_.value()),
          std::move(layout_optimizer_dags_),
          std::move(rescaling_pass_dag_.value()),
          std::move(leveling_optimizer_dags_),
          std::move(ct_op_optimizer_dags_)};
}

template <>
void WriteStream<CompiledProgram>(std::ostream& stream,
                                  const CompiledProgram& cp);

template <>
CompiledProgram ReadStream<CompiledProgram>(std::istream& stream);

}  // namespace fhelipe

#endif  // FHELIPE_COMPILED_PROGRAM_H_
