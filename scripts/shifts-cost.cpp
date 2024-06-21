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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <vector>

constexpr uint32_t s_mask(uint32_t s) { return (1U << s) - 1U; }

struct Bitset {
  static constexpr uint32_t bShift = 5;
  static constexpr uint32_t bSize = 1U << 5;
  static constexpr uint32_t bMask = s_mask(bShift);

  std::vector<uint32_t> bits;
  void resize(uint32_t n) { bits.resize(n >> bShift); }

  bool push(uint32_t i) {
    uint32_t& b = bits[i >> bShift];
    uint32_t si = 1 << (i & bMask);
    bool r = !(b & si);
    b |= si;
    return r;
  }
  void pop(uint32_t i) { bits[i >> bShift] = 0; }
};

using Layout = std::vector<uint32_t>;

uint32_t permute(const Layout& l, uint32_t x) {
  uint32_t r = 0;
  for (uint32_t i = 0; i < l.size(); ++i) {
    r |= ((x >> i) & 1U) << l[i];
  }
  return r;
}

using LayoutMap = std::vector<uint16_t>;

// 64 Ki
void build_layout_map(const Layout& l, LayoutMap& lm) {
  lm.resize(1U << l.size());
  for (uint32_t i = 0; i < lm.size(); i++) {
    lm[i] = permute(l, i);
  }
}

uint32_t layout_lookup(const LayoutMap& lm, uint16_t x) {
  uint16_t mask = lm.size() - 1;
  return x & (~mask) | lm[x & mask];
}

uint32_t carry_offset(const LayoutMap& lm, uint16_t c) {
  uint32_t c_add = layout_lookup(lm, c);
  uint32_t c_sub = layout_lookup(lm, c >> 1U) << 1U;
  uint32_t layout_mask = lm.size() - 1;
  return (c_add - c_sub) & layout_mask;
}

const std::vector<uint32_t>& shift_carries(uint32_t shift, uint32_t start_i,
                                           uint32_t end_i) {
  static std::vector<uint32_t> carries[2];
  carries[0].resize(1);
  carries[1].clear();

  if (start_i > 0) {
    carries[1].push_back(1 << start_i);
  }
  for (int i = start_i; i < end_i; i++) {
    if ((shift >> i) & 1) {
      carries[1].insert(carries[1].end(), carries[0].begin(), carries[0].end());
    } else {
      carries[0].insert(carries[0].end(), carries[1].begin(), carries[1].end());
    }
    for (uint32_t& c : carries[1]) {
      c |= 1U << (i + 1);
    }
  }
  carries[0].insert(carries[0].end(), carries[1].begin(), carries[1].end());
  return carries[0];
}

uint32_t shift_stage(const LayoutMap& lm, uint32_t shift, uint32_t start_i,
                     uint32_t end_i) {
  static Bitset bs;
  static std::vector<uint32_t> offsets;

  bs.resize(lm.size());

  auto& carries = shift_carries(shift, start_i, end_i);
  offsets.resize(carries.size());

  uint32_t rotates = 0;
  for (uint32_t i = 0; i < carries.size(); ++i) {
    offsets[i] = carry_offset(lm, carries[i]);
    rotates += bs.push(offsets[i]);
  }

  for (uint32_t i = 0; i < carries.size(); ++i) {
    bs.pop(offsets[i]);
  }

  return rotates;
}

struct ShiftResult {
  uint32_t stages = 0;
  uint32_t rotates = 0;
};

ShiftResult run_shift(const LayoutMap& lm, uint32_t shift, uint32_t max_g) {
  ShiftResult res;
  uint32_t log_n = std::__bit_width(lm.size()) - 1;
  assert(lm.size() == (1U << log_n));
  for (uint32_t i = 0; i < log_n;) {
    uint32_t prev_rotates = 0;

    for (uint32_t j = i + 1; j <= log_n; ++j) {
      uint32_t rotates = shift_stage(lm, shift, i, j);
      if (rotates > max_g) {
        res.rotates += prev_rotates;
        res.stages++;
        prev_rotates = 0;
        i = j - 1;
        break;
      }
      prev_rotates = rotates;
    }

    // Got to the end
    if (prev_rotates > 0) {
      res.rotates += prev_rotates;
      res.stages++;
      break;
    }
  }
  return res;
}

double cost_lower_bound(uint32_t rotates, uint32_t stages) {
  double stage_work = std::pow(double(rotates), 1.0 / stages);
  // Should we ceil here?
  return stage_work * stages;
}

double fhelipe_overheads(const LayoutMap& lm, uint32_t shift) {
  constexpr uint32_t fhelipe_max_g = 16;

  uint32_t rotates = run_shift(lm, shift, lm.size()).rotates;
  ShiftResult fhelipe_res = run_shift(lm, shift, fhelipe_max_g);

  uint32_t lb = cost_lower_bound(rotates, fhelipe_res.stages);
  double r = double(fhelipe_res.rotates) / lb;

  // if (r > 5.0) {
  //     std::cerr << "Shift: " << shift << "\n";
  //     std::cerr << "Fhelipe " << fhelipe_res.rotates << ", " <<
  //     fhelipe_res.stages << "\n"; std::cerr << "Rg: " << rotates << " lb: "
  //     << lb << "\n"; std::cerr << "Ratio " << (double(fhelipe_res.rotates)) /
  //     lb << "\n\n";
  // }
  return r;
}

struct GMeanAcc {
  std::vector<std::optional<double>> v;

  void push(double x) {
    for (auto& y : v) {
      if (y.has_value()) {
        x = std::sqrt(x * (*y));
        y = std::nullopt;
      } else {
        y = x;
        return;
      }
    }
    v.push_back(x);
  }

  double get() const {
    uint64_t cnt = 0;
    double prod = 1;
    for (uint32_t i = 0; i < v.size(); i++) {
      if (v[i]) {
        prod = prod * (*v[i]);
        cnt += 1U << i;
      }
      prod = std::sqrt(prod);
    }
    // Prod has all numbers to the power 2^(v.size())
    double exp = double(1ULL << v.size()) / cnt;
    return std::pow(prod, exp);
  }
};

struct StatAcc {
  double sum = 0;
  double sq_sum = 0;
  double max = 0;
  uint64_t cnt = 0;

  void push(double r);

  double avg() const { return sum / cnt; }
  double std_dev() const { return std::sqrt(sq_sum / cnt - avg() * avg()); }

  double nth(uint64_t n) const {
    std::nth_element(nums.begin(), nums.begin() + n, nums.end());
    return nums[n];
  }

 private:
  mutable std::vector<double> nums;
};

std::ostream& operator<<(std::ostream& s, const StatAcc& acc) {
  s << "{";
  s << "mean: " << acc.avg();
  s << " std_dev: " << acc.std_dev();
  s << " max: " << acc.max << "\n";
  for (uint32_t i = 0; i < 20; ++i) {
    s << "    " << i * 100 / 20 << "%: " << acc.nth(acc.cnt * i / 20) << "\n";
    // s << " 50%: " << acc.nth(acc.cnt / 2);
    // s << " 75%: " << acc.nth(acc.cnt * 3 / 4);
  }
  return s << "}";
}

void StatAcc::push(double r) {
  nums.push_back(r);
  sum += r;
  sq_sum += r * r;
  max = std::max(max, r);
  ++cnt;
}

void eval_layout(const Layout& l, StatAcc& acc) {
  static LayoutMap lm;
  build_layout_map(l, lm);

  for (uint32_t s = 1; s * 2 <= lm.size(); ++s) {
    double r = fhelipe_overheads(lm, s);
    acc.push(r);
  }
}

Layout init_layout(uint32_t log_n) {
  Layout l(log_n);
  for (uint32_t i = 0; i < log_n; ++i) {
    l[i] = i;
  }
  return l;
}

StatAcc eval_all_layouts(uint32_t log_n) {
  std::cout << "Exhaustive n=" << (1U << log_n) << "\n";
  StatAcc acc;

  Layout l = init_layout(log_n);
  do {
    eval_layout(l, acc);
  } while (std::next_permutation(l.begin(), l.end()));

  return acc;
}

StatAcc eval_random_layouts(uint32_t log_n, uint32_t task_i) {
  constexpr uint64_t sample_cnt = 10'000;
  std::mt19937 rng(0xCAFEFACE ^ task_i);

  std::cout << "Random sample n=" << (1U << log_n) << " samples=" << sample_cnt
            << "\n";

  StatAcc acc;
  Layout l = init_layout(log_n);
  for (uint32_t i = 0; i < sample_cnt; i++) {
    std::shuffle(l.begin(), l.end(), rng);
    eval_layout(l, acc);
    if (i % 500 == 0 && i > 0) {
      std::cerr << i << ": " << acc << "\n";
    }
  }

  return acc;
}

int main(int argc, char** argv) {
  assert(argc >= 2);
  uint32_t log_n = std::stoull(argv[1]);
  uint32_t task_i = (argc >= 3) ? std::stoull(argv[2]) : 0;

  StatAcc res;
  if (log_n < 10) {
    res = eval_all_layouts(log_n);
  } else {
    res = eval_random_layouts(log_n, task_i);
  }

  std::cout << res << "\n";
}
