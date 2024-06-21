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

#include "include/fhebooster_pass.h"
#include <fstream>
#include <iostream>
#include <limits>

#include "include/dag.h"
#include "include/ct_program.h"
#include "include/extended_std.h"
#include "include/level.h"
#include "include/mul_cc.h"
#include "include/mul_cp.h"
#include "include/io_utils.h"
#include "include/io_spec.h"
#include "include/dag_io.h"
#include "include/rescale_c.h"

namespace fhelipe {

namespace {

    /*
std::ostream&
operator<<( std::ostream& dest, __int128_t value )
{
    std::ostream::sentry s( dest );
    if ( s ) {
        __uint128_t tmp = value < 0 ? -value : value;
        char buffer[ 128 ];
        char* d = std::end( buffer );
        do
        {
            -- d;
            *d = "0123456789"[ tmp % 10 ];
            tmp /= 10;
        } while ( tmp != 0 );
        if ( value < 0 ) {
            -- d;
            *d = '-';
        }
        int len = std::end( buffer ) - d;
        if ( dest.rdbuf()->sputn( d, len ) != len ) {
            dest.setstate( std::ios_base::badbit );
        }
    }
    return dest;
}
*/

std::unordered_map<std::shared_ptr<Node<CtOp>>, std::vector<__int128>> backward_path_count;
std::unordered_map<std::shared_ptr<Node<CtOp>>, std::vector<__int128>> forward_path_count;
std::unordered_map<std::shared_ptr<Node<CtOp>>, __int128> path_counts;
std::set<std::shared_ptr<Node<CtOp>>> already_bootstrapped;

void UpdatePathCounts(const Dag<CtOp>& out_dag, int usable_levels) {
    for (const auto& node : out_dag.NodesInTopologicalOrder()) {
        path_counts[node] = 0;
        if (Estd::contains(already_bootstrapped, node)) {
            continue;
        }
        for (auto lvl : Estd::indices(usable_levels)) {
            // std::cout << "backward[" << node->NodeId() << "][" << lvl << "] = " << backward_path_count[node][lvl] << std::endl;
            // std::cout << "forward[" << node->NodeId() << "][" << lvl << "] = " << forward_path_count[node][lvl] << std::endl;
            path_counts[node] += (backward_path_count[node][lvl] * forward_path_count[node][usable_levels - 1 - lvl]);
            if (path_counts[node] < 0) {
                std::cout << "OVERFLOW" << std::endl;
                path_counts[node] = std::numeric_limits<__int128>::max();
                break;
            }
        }
    }
    /*
    for (const auto& node : out_dag.NodesInTopologicalOrder()) {
      std::cout << node->NodeId() << ": " << path_counts[node] << std::endl;
    }
    */
}

void UpdateForwardAndBackward(const Dag<CtOp>& out_dag, int usable_levels) {
    for (const auto& node : out_dag.NodesInTopologicalOrder()) {
        backward_path_count[node] = std::vector<__int128>(usable_levels, 0);
        if (Estd::contains(already_bootstrapped, node)) {
          continue;
        }
        if (dynamic_cast<const RescaleC*>(&node->Value())) {
          backward_path_count[node][0] = 1;
        }
        for (auto lvl : Estd::indices(usable_levels)) {
            for (auto parent : Estd::vector_to_set(node->Parents())) {
              if (dynamic_cast<const RescaleC*>(&node->Value())) {
                  if (lvl > 0) {
                    backward_path_count[node][lvl] += backward_path_count[parent][lvl-1];
                  }
              } else {
                backward_path_count[node][lvl] += backward_path_count[parent][lvl];
              }
            }
        }
    }

    for (const auto& node : out_dag.NodesInReverseTopologicalOrder()) {
        forward_path_count[node] = std::vector<__int128>(usable_levels, 0);
        if (Estd::contains(already_bootstrapped, node)) {
          continue;
        }
        if (Estd::any_of(node->Children(), [](const auto& child) { return dynamic_cast<RescaleC*>(&child->Value()); })) {
          forward_path_count[node][0] = 1;
        }
        for (auto lvl : Estd::indices(usable_levels)) {
            for (auto child : node->Children()) {
               if (dynamic_cast<const RescaleC*>(&node->Value())) {
                if (lvl > 0) {
                  forward_path_count[node][lvl] += forward_path_count[child][lvl-1];
                }
               } else {
                if (!(lvl == 0 && Estd::any_of(node->Children(), [](const auto& child) { return dynamic_cast<RescaleC*>(&child->Value()); }))) {
                    forward_path_count[node][lvl] += forward_path_count[child][lvl];
                }
              }
            }
        }
        // std::cout << "children: " << node->Children().size() << "; " << forward_path_count[node][0] << std::endl;
    }
}

std::shared_ptr<Node<CtOp>> FindMaxNode(const std::unordered_map<std::shared_ptr<Node<CtOp>>, __int128> path_counts) {
    std::shared_ptr<Node<CtOp>> max_node;
    __int128 max_value = -1;
    for (const auto& [key, value] : path_counts) {
        if (max_value < value) {
            max_node = key;
            max_value = value;
        }
    }
    return max_node;
}

}



CtOpOptimizerOutput FheBoosterPass::DoPass(const CtOpOptimizerInput& in_dag) {
    std::ofstream f("/tmp/fhebooster.txt");
    f << "Hello" << std::endl;
    auto out_dag = CloneFromAncestor(in_dag.GetDag());


    UpdateForwardAndBackward(out_dag, usable_levels_.value());
    UpdatePathCounts(out_dag, usable_levels_.value());


    int boot_count = 0;
    __int128 max_value = 1;
    std::shared_ptr<Node<CtOp>> max_node;
    while (max_value > 0) {
        max_node = FindMaxNode(path_counts);
        max_value = path_counts[max_node];
        // std::cout << "path_counts.size(): " << path_counts.size() << std::endl;
        // std::cout << "max_value: " << max_value << std::endl;
        boot_count++;
        // std::cout << "bootstrapped: " << already_bootstrapped.size() << " / " << path_counts.size() << std::endl;

        already_bootstrapped.insert(max_node);

        UpdateForwardAndBackward(out_dag, usable_levels_.value());
        UpdatePathCounts(out_dag, usable_levels_.value());
    }

    std::cout << "boot count: " << boot_count << std::endl;

    return ct_program::CtProgram{in_dag.GetProgramContext(),
                                 in_dag.ChunkDictionary()->CloneUniq(),
                                 std::move(out_dag)};
}

}

