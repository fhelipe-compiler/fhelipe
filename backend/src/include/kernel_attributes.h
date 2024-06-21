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

#ifndef FHELIPE_KERNEL_ATTRIBUTES_H_
#define FHELIPE_KERNEL_ATTRIBUTES_H_

#include <vector>

#include "shape.h"

namespace fhelipe {

class KernelAttributes {
 public:
  // `kernel_shape` includes only the spacial dimensions of the kernel,
  //                i.e, it must have the same number of dimensions as
  //                `stride` and `padding`.
  // `stride` defaults to `1` along all dimensions
  // `padding` defaults to `0` along all dimensions
  KernelAttributes(const Shape& kernel_shape,
                   const std::vector<int>& strides = {},
                   const std::vector<int>& pads = {});

  int DimensionCount() const;
  const Shape& KernelShape() const;
  const std::vector<int>& Strides() const;
  const std::vector<int>& BeginPads() const;
  const std::vector<int>& EndPads() const;

  Shape OutputShape(const Shape& input_shape, int output_channels) const;
  Shape OutputShape(const Shape& Shape) const;

 private:
  Shape kernel_shape_;
  std::vector<int> strides_;
  std::vector<int> begin_pads_;
  std::vector<int> end_pads_;
};

}  // namespace fhelipe

#endif  // FHELIPE_KERNEL_ATTRIBUTES_H_
