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

#include "include/level_info_utils.h"

#include "include/extended_std.h"
#include "include/level_info.h"
#include "include/program_context.h"

namespace fhelipe {

namespace {

LevelInfo WaterlineRescale(LogScale log_eva_s_w, LogScale log_eva_s_f,
                           const LevelInfo& level_info) {
  Level level = level_info.Level();
  LogScale log_scale = level_info.LogScale();
  while (log_scale >= log_eva_s_w + log_eva_s_f) {
    log_scale = log_scale - log_eva_s_f;
    level = level - Level(1);
  }
  return {level, log_scale};
}

}  // namespace

LevelInfo LevelInfoForAddCC(const ProgramContext& context,
                            const LevelInfo& parent_1,
                            const LevelInfo& parent_2) {
  Level level = std::min(parent_1.Level(), parent_2.Level());
  LogScale log_scale =
      LogScale(std::max(parent_1.LogScale(), parent_2.LogScale()));

  return WaterlineRescale(context.LogScale(), context.LogScale(),
                          LevelInfo{level, log_scale});
}

LevelInfo LevelInfoForMulCC(const ProgramContext& context,
                            const LevelInfo& parent_1,
                            const LevelInfo& parent_2) {
  Level level = std::min(parent_1.Level(), parent_2.Level());
  LogScale log_scale = parent_1.LogScale() + parent_2.LogScale();
  return WaterlineRescale(context.LogScale(), context.LogScale(),
                          LevelInfo{level, log_scale});
}  // namespace fhelipe

LevelInfo LevelInfoForMulCS(const ProgramContext& context,
                            const LevelInfo& parent) {
  return LevelInfoForMulCP(context, parent);
}

LevelInfo LevelInfoForAddCS(const ProgramContext& context,
                            const LevelInfo& parent) {
  return LevelInfoForAddCP(context, parent);
}

LevelInfo LevelInfoForMulCP(const ProgramContext& context,
                            const LevelInfo& parent) {
  // Assumes plaintext is at the same scale as the ciphertext
  return WaterlineRescale(context.LogScale(), context.LogScale(),
                          LevelInfo{parent.Level(), 2 * parent.LogScale()});
}

LevelInfo LevelInfoForAddCP(const ProgramContext& context,
                            const LevelInfo& parent) {
  // Assumes plaintext is at the same scale as the ciphertext
  (void)context;
  return parent;
}

LevelInfo LevelInfoForRotateC(const ProgramContext& context,
                              const LevelInfo& parent) {
  (void)context;
  return parent;
}

LevelInfo LevelInfoForBackendMask(const ProgramContext& context,
                                  const LevelInfo& parent) {
  // Assumes mask is at the same scale as the ciphertext
  return WaterlineRescale(context.LogScale(), context.LogScale(),
                          LevelInfo{parent.Level(), 2 * parent.LogScale()});
}

}  // namespace fhelipe
