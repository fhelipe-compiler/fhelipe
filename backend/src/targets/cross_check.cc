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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "include/checker.h"
#include "include/persisted_dictionary.h"
#include "include/tensor.h"
#include "targets/gflag_utils/exe_folder_gflag_utils.h"
#include "targets/gflag_utils/verbose_gflag_utils.h"

using namespace fhelipe;

DEFINE_bool(no_complain, false,
            "Set flag to run checker on all outputs, without breaking if one "
            "of them is not `CloseEnough'");
DEFINE_bool(all_diffs, false,
            "Set flag to get a csv of the absolute value diffs of double vs "
            "encrypted computation.");
DEFINE_string(check_me_only, "", "Set to only check that output");

int main(int argc, char* argv[]) {
  double epsilon = 1e-2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::ios_base::sync_with_stdio(false);

  auto exe_folder = ExeFolderFromFlags();
  auto checks = PersistedDictionary<Tensor<PtVal>>(exe_folder / kOutCheck);
  auto outputs = PersistedDictionary<Tensor<PtVal>>(exe_folder / kOutUnenc);
  CHECK(!checks.Keys().empty());
  if (FLAGS_verbose) {
    std::cout << "Name,LogMaxError,MaxErrorIdx" << std::endl;
  }
  auto keys = FLAGS_check_me_only.empty()
                  ? checks.Keys()
                  : std::set<std::string>{FLAGS_check_me_only};
  for (const auto& key : keys) {
    auto check_tensor = checks.At(key);
    auto output_tensor = outputs.At(key);
    CHECK(check_tensor.GetShape() == output_tensor.GetShape());
    if (!FLAGS_no_complain) {
      TestCloseEnough(check_tensor.Values(), output_tensor.Values(), epsilon);
    }
    if (FLAGS_verbose) {
      auto diff = Estd::transform(check_tensor.Values(), output_tensor.Values(),
                                  std::minus<>());
      std::cout << key << "," << std::log2(LInfinityNorm(diff)) << ","
                << Estd::argmax(Estd::transform(
                       diff, [](const auto& value) { return std::abs(value); }))
                << std::endl;
    }
    if (FLAGS_all_diffs) {
      if (key == "result") {
        auto diffs = Estd::transform(
            check_tensor.Values(), output_tensor.Values(),
            [](const auto& x, const auto& y) { return std::abs(x - y); });
        for (auto diff : diffs) {
          std::cout << std::log2(diff) << " ";
        }
      }
    }
  }
  CHECK(checks.Keys() == outputs.Keys());
}
