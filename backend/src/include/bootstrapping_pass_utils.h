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

#ifndef FHELIPE_BOOTSTRAPPING_PASS_UTILS_H_
#define FHELIPE_BOOTSTRAPPING_PASS_UTILS_H_

#include <set>

#include "level.h"
#include "node.h"

namespace fhelipe {

class Level;
class LevelInfo;
class LeveledTOp;
class ScaledTOp;

static const Level kMinLevel = 1;

Level MinLevel(const std::set<const LeveledTOp*>& ops);

std::vector<LevelInfo> ExtractLevelInfos(
    const std::vector<std::shared_ptr<Node<LeveledTOp>>>& nodes);

LevelInfo NodeLevelInfo(const Node<ScaledTOp>& scaled_node,
                        const std::vector<LevelInfo>& parents_level_info,
                        Level max_usable_levels);

}  // namespace fhelipe

#endif  // FHELIPE_BOOTSTRAPPING_PASS_UTILS_H_
