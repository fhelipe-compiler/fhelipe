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

#include "include/compiled_program.h"

#include "include/dag.h"
#include "include/debug_info.h"
#include "include/pass_utils.h"

namespace fhelipe {

CompiledProgram::CompiledProgram(
    const Program& program, std::pair<PassId, ParserOutput>&& input_dag,
    std::vector<std::pair<PassId, LayoutOptimizerOutput>>&&
        layout_optimizer_dags,
    std::pair<PassId, RescalingPassOutput>&& rescaling_pass_dag,
    std::vector<std::pair<PassId, LevelingPassOutput>>&&
        leveling_optimizer_dags,
    std::vector<std::pair<PassId, CtOpOptimizerOutput>>&& ct_op_optimizer_dags)
    : program_(program),
      input_dag_(std::move(input_dag)),
      layout_optimizer_dags_(std::move(layout_optimizer_dags)),
      rescaling_pass_dag_(std::move(rescaling_pass_dag)),
      leveling_optimizer_dags_(std::move(leveling_optimizer_dags)),
      ct_op_optimizer_dags_(std::move(ct_op_optimizer_dags)) {
  archives_.emplace_back(input_dag_.first,
                         DagToDebugInfoArchive(input_dag_.second));
  for (const auto& optimizer : layout_optimizer_dags_) {
    archives_.emplace_back(optimizer.first,
                           DagToDebugInfoArchive(optimizer.second));
  }
  archives_.emplace_back(rescaling_pass_dag_.first,
                         DagToDebugInfoArchive(rescaling_pass_dag_.second));
  for (const auto& optimizer : leveling_optimizer_dags_) {
    archives_.emplace_back(optimizer.first,
                           DagToDebugInfoArchive(optimizer.second));
  }
  for (const auto& optimizer : ct_op_optimizer_dags_) {
    archives_.emplace_back(optimizer.first,
                           DagToDebugInfoArchive(optimizer.second.GetDag()));
  }
}

std::vector<PassId> CompiledProgram::GetPassIds() const {
  std::vector<PassId> result;
  for (const auto& archive : archives_) {
    result.push_back(archive.first);
  }
  return result;
}

template <>
void WriteStream<PassId>(std::ostream& stream, const PassId& pass_id) {
  WriteStream<std::string>(stream, PaddedIntString(pass_id.PassIndex()) + "_");
  WriteStream<PassName>(stream, pass_id.GetPassName());
}

template <>
PassId ReadStream<PassId>(std::istream& stream) {
  std::string pass_id_str = ReadStream<std::string>(stream);
  std::stringstream ss;
  ss << pass_id_str;
  std::string segment;
  std::getline(ss, segment, '_');
  int idx = std::atoi(segment.c_str());
  PassName pass_name = ReadStream<PassName>(ss);
  return PassId{idx, pass_name};
}

std::string ToString(const PassId& pass_id) {
  std::stringstream ss;
  WriteStream(ss, pass_id);
  return ss.str();
}

void WriteStreamFriend(std::ostream& stream, const CompiledProgram& cp) {
  const auto& exe_folder = cp.ExeFolder();
  WriteStream(stream, exe_folder);
  stream << " ";

  WriteStream(stream, cp.layout_optimizer_dags_.size());
  stream << " ";
  WriteStream(stream, cp.leveling_optimizer_dags_.size());
  stream << " ";
  WriteStream(stream, cp.ct_op_optimizer_dags_.size());
  stream << " ";

  WriteFile(exe_folder / kDslOutput, cp.FrontendCode());

  WriteStream(stream, cp.GetPassIds());

  WriteFile(exe_folder / ToString(cp.input_dag_.first), cp.input_dag_.second);

  for (const auto& [pass_id, dag] : cp.layout_optimizer_dags_) {
    WriteFile(exe_folder / ToString(pass_id), dag);
  }

  WriteFile(exe_folder / ToString(cp.rescaling_pass_dag_.first),
            cp.rescaling_pass_dag_.second);

  for (const auto& [pass_id, dag] : cp.leveling_optimizer_dags_) {
    WriteFile(exe_folder / ToString(pass_id), dag);
  }

  for (const auto& [pass_id, dag] : cp.ct_op_optimizer_dags_) {
    WriteFile(exe_folder / ToString(pass_id), dag);
  }
}

std::string PaddedIntString(int value) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(3) << value;
  return ss.str();
}

template <>
CompiledProgram ReadStream<CompiledProgram>(std::istream& stream) {
  const auto exe_folder = ReadStream<std::filesystem::path>(stream);
  auto program = ReadFile<Program>(exe_folder / kDslOutput);
  auto layout_pass_count = ReadStream<int>(stream);
  auto leveling_pass_count = ReadStream<int>(stream);
  auto ct_op_pass_count = ReadStream<int>(stream);
  auto pass_ids = ReadStream<std::vector<PassId>>(stream);

  std::vector<std::pair<PassId, LayoutOptimizerOutput>> layout_optimizer_dags;
  std::vector<std::pair<PassId, LevelingOptimizerOutput>>
      leveling_optimizer_dags;
  std::vector<std::pair<PassId, CtOpOptimizerOutput>> ct_op_optimizer_dags;

  int pass_count = 0;
  auto input_dag_pass_id = pass_ids.at(pass_count++);
  std::pair<PassId, ParserOutput> input_dag{
      input_dag_pass_id,
      ReadFile<ParserOutput>(exe_folder / ToString(input_dag_pass_id))};

  for (; pass_count < 1 + layout_pass_count; ++pass_count) {
    layout_optimizer_dags.emplace_back(
        pass_ids.at(pass_count),
        ReadFile<LayoutOptimizerOutput>(exe_folder /
                                        ToString(pass_ids.at(pass_count))));
  }
  auto rescaling_dag_pass_id = pass_ids.at(pass_count++);
  std::pair<PassId, RescalingPassOutput> rescaling_pass_dag{
      rescaling_dag_pass_id, ReadFile<RescalingPassOutput>(
                                 exe_folder / ToString(rescaling_dag_pass_id))};
  for (; pass_count < 2 + layout_pass_count + leveling_pass_count;
       ++pass_count) {
    leveling_optimizer_dags.emplace_back(
        pass_ids.at(pass_count),
        ReadFile<LevelingOptimizerOutput>(exe_folder /
                                          ToString(pass_ids.at(pass_count))));
  }
  for (; pass_count <
         (2 + layout_pass_count + leveling_pass_count + ct_op_pass_count);
       ++pass_count) {
    ct_op_optimizer_dags.emplace_back(
        pass_ids.at(pass_count),
        ReadFile<CtOpOptimizerOutput>(exe_folder /
                                      ToString(pass_ids.at(pass_count))));
  }
  return {program,
          std::move(input_dag),
          std::move(layout_optimizer_dags),
          std::move(rescaling_pass_dag),
          std::move(leveling_optimizer_dags),
          std::move(ct_op_optimizer_dags)};
}

DebugInfoArchive CompiledProgram::GetDebugInfoArchive(
    const PassName& source_pass, const PassName& destination_pass) const {
  std::vector<DebugInfoArchive> archive_list =
      GetDebugInfoArchivesBetween(source_pass, destination_pass);
  DebugInfoArchive merged = archive_list.back();
  archive_list.pop_back();
  while (!archive_list.empty()) {
    merged = MergeAdjacent(archive_list.back(), merged);
    archive_list.pop_back();
  }
  return merged;
}

int CompiledProgram::FindPassIndex(const PassName& pass_name) const {
  for (int idx = 0; idx < archives_.size(); ++idx) {
    if (archives_.at(idx).first.GetPassName() == pass_name) {
      return idx;
    }
  }
  LOG(FATAL);
}

const CtOpOptimizerOutput& CompiledProgram::Result() const {
  return ct_op_optimizer_dags_.back().second;
}

std::vector<DebugInfoArchive> CompiledProgram::GetDebugInfoArchivesBetween(
    const PassName& source_pass, const PassName& destination_pass) const {
  std::vector<DebugInfoArchive> result;
  int source_idx = FindPassIndex(source_pass);
  int destination_idx = FindPassIndex(destination_pass);
  CHECK(source_idx < destination_idx);
  for (int idx = source_idx + 1; idx <= destination_idx; ++idx) {
    result.push_back(archives_.at(idx).second);
  }
  return result;
}

}  // namespace fhelipe
