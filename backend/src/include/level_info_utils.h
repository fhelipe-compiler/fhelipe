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

#ifndef FHELIPE_LEVEL_INFO_UTILS_H_
#define FHELIPE_LEVEL_INFO_UTILS_H_

#include "latticpp/ckks/lattigo_param.h"
#include "level_info.h"
#include "program_context.h"

namespace fhelipe {

LevelInfo LevelInfoForAddCC(const ProgramContext& context,
                            const LevelInfo& parent_1,
                            const LevelInfo& parent_2);
LevelInfo LevelInfoForMulCC(const ProgramContext& context,
                            const LevelInfo& parent_1,
                            const LevelInfo& parent_2);
LevelInfo LevelInfoForMulCP(const ProgramContext& context,
                            const LevelInfo& parent);
LevelInfo LevelInfoForAddCP(const ProgramContext& context,
                            const LevelInfo& parent);
LevelInfo LevelInfoForMulCS(const ProgramContext& context,
                            const LevelInfo& parent);
LevelInfo LevelInfoForAddCS(const ProgramContext& context,
                            const LevelInfo& parent);
LevelInfo LevelInfoForRotateC(const ProgramContext& context,
                              const LevelInfo& parent);
LevelInfo LevelInfoForBackendMask(const ProgramContext& context,
                                  const LevelInfo& parent);

}  // namespace fhelipe

#endif  // FHELIPE_LEVEL_INFO_UTILS_H_
