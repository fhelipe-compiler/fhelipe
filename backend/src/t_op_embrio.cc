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

#include "include/t_op_embrio.h"

#include "include/constants.h"
#include "include/io_utils.h"
#include "include/t_add_cc.h"

namespace fhelipe {

DerivedRegistrar<TAddCCEmbrio> TAddCCEmbrio::reg_{
    TAddCCEmbrio::StaticTypeName()};
DerivedRegistrar<TAddCPEmbrio> TAddCPEmbrio::reg_{
    TAddCPEmbrio::StaticTypeName()};
DerivedRegistrar<TAddCSIEmbrio> TAddCSIEmbrio::reg_{
    TAddCSIEmbrio::StaticTypeName()};
DerivedRegistrar<TChetRepackCEmbrio> TChetRepackCEmbrio::reg_{
    TChetRepackCEmbrio::StaticTypeName()};

DerivedRegistrar<TMulCCEmbrio> TMulCCEmbrio::reg_{
    TMulCCEmbrio::StaticTypeName()};
DerivedRegistrar<TMulCPEmbrio> TMulCPEmbrio::reg_{
    TMulCPEmbrio::StaticTypeName()};
DerivedRegistrar<TMulCSIEmbrio> TMulCSIEmbrio::reg_{
    TMulCSIEmbrio::StaticTypeName()};

DerivedRegistrar<TInputCEmbrio> TInputCEmbrio::reg_{
    TInputCEmbrio::StaticTypeName()};
DerivedRegistrar<TOutputCEmbrio> TOutputCEmbrio::reg_{
    TOutputCEmbrio::StaticTypeName()};

DerivedRegistrar<TReduceDimCEmbrio> TReduceDimCEmbrio::reg_{
    TReduceDimCEmbrio::StaticTypeName()};
DerivedRegistrar<TReorderDimsCEmbrio> TReorderDimsCEmbrio::reg_{
    TReorderDimsCEmbrio::StaticTypeName()};
DerivedRegistrar<TReplicateDimCEmbrio> TReplicateDimCEmbrio::reg_{
    TReplicateDimCEmbrio::StaticTypeName()};
DerivedRegistrar<TDropDimCEmbrio> TDropDimCEmbrio::reg_{
    TDropDimCEmbrio::StaticTypeName()};
DerivedRegistrar<TInsertDimCEmbrio> TInsertDimCEmbrio::reg_{
    TInsertDimCEmbrio::StaticTypeName()};
DerivedRegistrar<TResizeDimCEmbrio> TResizeDimCEmbrio::reg_{
    TResizeDimCEmbrio::StaticTypeName()};
DerivedRegistrar<TStrideCEmbrio> TStrideCEmbrio::reg_{
    TStrideCEmbrio::StaticTypeName()};
DerivedRegistrar<TMergedStrideCEmbrio> TMergedStrideCEmbrio::reg_{
    TMergedStrideCEmbrio::StaticTypeName()};
DerivedRegistrar<TUnpaddedShiftCEmbrio> TUnpaddedShiftCEmbrio::reg_{
    TUnpaddedShiftCEmbrio::StaticTypeName()};
DerivedRegistrar<TCyclicShiftCEmbrio> TCyclicShiftCEmbrio::reg_{
    TCyclicShiftCEmbrio::StaticTypeName()};
DerivedRegistrar<TBootstrapCEmbrio> TBootstrapCEmbrio::reg_{
    TBootstrapCEmbrio::StaticTypeName()};
DerivedRegistrar<TRotateCEmbrio> TRotateCEmbrio::reg_{
    TRotateCEmbrio::StaticTypeName()};

template <>
void WriteStream<TRotateCEmbrio>(std::ostream& stream,
                                 const TRotateCEmbrio& node) {
  stream << TRotateCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " 0 ";
  WriteStream<int>(stream, node.RotateBy());
}

template <>
TRotateCEmbrio ReadStreamWithoutTypeNamePrefix<TRotateCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto trash = ReadStream<int>(stream);
  CHECK(trash == 0);
  auto rotate_by = ReadStream<int>(stream);
  return TRotateCEmbrio{shape, rotate_by};
}

template <>
void WriteStream<TChetRepackCEmbrio>(std::ostream& stream,
                                     const TChetRepackCEmbrio& node) {
  stream << TChetRepackCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
}

template <>
TChetRepackCEmbrio ReadStreamWithoutTypeNamePrefix<TChetRepackCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  return TChetRepackCEmbrio{shape};
}

template <>
void WriteStream<TAddCCEmbrio>(std::ostream& stream, const TAddCCEmbrio& node) {
  stream << TAddCCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
}

template <>
TAddCCEmbrio ReadStreamWithoutTypeNamePrefix<TAddCCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  return TAddCCEmbrio{shape};
}

template <>
void WriteStream<TAddCPEmbrio>(std::ostream& stream, const TAddCPEmbrio& node) {
  stream << TAddCPEmbrio::StaticTypeName() << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<std::string>(stream, node.PtTensorName());
  stream << " ";
  WriteStream<LogScale>(stream, node.PtTensorLogScale());
}

template <>
TAddCPEmbrio ReadStreamWithoutTypeNamePrefix<TAddCPEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto pt_tensor_name = ReadStream<std::string>(stream);
  auto pt_tensor_log_scale = ReadStream<LogScale>(stream);
  return {shape, pt_tensor_name, pt_tensor_log_scale};
}

template <>
void WriteStream<TMulCCEmbrio>(std::ostream& stream, const TMulCCEmbrio& node) {
  stream << TMulCCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
}

template <>
TMulCCEmbrio ReadStreamWithoutTypeNamePrefix<TMulCCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  return TMulCCEmbrio{shape};
}

template <>
void WriteStream<TMulCPEmbrio>(std::ostream& stream, const TMulCPEmbrio& node) {
  stream << TMulCPEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<std::string>(stream, node.PtTensorName());
  stream << " ";
  WriteStream<LogScale>(stream, node.PtTensorLogScale());
}

template <>
TMulCPEmbrio ReadStreamWithoutTypeNamePrefix<TMulCPEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto pt_tensor_name = ReadStream<std::string>(stream);
  auto pt_tensor_log_scale = ReadStream<LogScale>(stream);
  return TMulCPEmbrio{shape, pt_tensor_name, pt_tensor_log_scale};
}

template <>
void WriteStream<TInputCEmbrio>(std::ostream& stream,
                                const TInputCEmbrio& node) {
  stream << TInputCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<std::string>(stream, node.TensorName());
  stream << " ";
  WriteStream<LogScale>(stream, node.GetLogScale());
}

template <>
TInputCEmbrio ReadStreamWithoutTypeNamePrefix<TInputCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto tensor_name = ReadStream<std::string>(stream);
  auto log_scale = ReadStream<LogScale>(stream);
  return {shape, tensor_name, log_scale};
}

template <>
void WriteStream<TOutputCEmbrio>(std::ostream& stream,
                                 const TOutputCEmbrio& node) {
  stream << TOutputCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<std::string>(stream, node.TensorName());
}

template <>
TOutputCEmbrio ReadStreamWithoutTypeNamePrefix<TOutputCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto tensor_name = ReadStream<std::string>(stream);
  return {shape, tensor_name};
}

template <>
void WriteStream<TReduceDimCEmbrio>(std::ostream& stream,
                                    const TReduceDimCEmbrio& node) {
  stream << TReduceDimCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<int>(stream, node.DimensionToReduce());
}

template <>
TReduceDimCEmbrio ReadStreamWithoutTypeNamePrefix<TReduceDimCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  return {shape, dim};
}

template <>
void WriteStream<TReorderDimsCEmbrio>(std::ostream& stream,
                                      const TReorderDimsCEmbrio& node) {
  stream << TReorderDimsCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream(stream, node.DimensionOrder());
}

template <>
TReorderDimsCEmbrio ReadStreamWithoutTypeNamePrefix<TReorderDimsCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim_order = ReadStream<std::vector<int>>(stream);
  return {shape, dim_order};
}

template <>
void WriteStream<TReplicateDimCEmbrio>(std::ostream& stream,
                                       const TReplicateDimCEmbrio& node) {
  stream << TReplicateDimCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<int>(stream, node.DimensionToReplicate());
  stream << " ";
  WriteStream<int>(stream, node.ReplicationMultiple());
}

template <>
TReplicateDimCEmbrio ReadStreamWithoutTypeNamePrefix<TReplicateDimCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  auto multiple = ReadStream<int>(stream);
  return {shape, dim, multiple};
}

template <>
void WriteStream<TDropDimCEmbrio>(std::ostream& stream,
                                  const TDropDimCEmbrio& node) {
  stream << TDropDimCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<int>(stream, node.DimensionToDrop());
}

template <>
TDropDimCEmbrio ReadStreamWithoutTypeNamePrefix<TDropDimCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  return {shape, dim};
}

template <>
void WriteStream<TInsertDimCEmbrio>(std::ostream& stream,
                                    const TInsertDimCEmbrio& node) {
  stream << TInsertDimCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<int>(stream, node.DimensionToInsert());
}

template <>
TInsertDimCEmbrio ReadStreamWithoutTypeNamePrefix<TInsertDimCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  return {shape, dim};
}

template <>
void WriteStream<TResizeDimCEmbrio>(std::ostream& stream,
                                    const TResizeDimCEmbrio& node) {
  stream << TResizeDimCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  const Shape& input_shape = node.InputShape();
  const Shape& output_shape = node.OutputShape();
  // Check that we are resizing at most one dimension
  CHECK(1 >= Estd::count_if(Estd::indices(input_shape.DimensionCount()),
                            [&input_shape, &output_shape](int idx) {
                              return input_shape[idx] != output_shape[idx];
                            }));
  for (int idx : Estd::indices(input_shape.DimensionCount())) {
    if (input_shape[idx] != output_shape[idx]) {
      WriteStream<int>(stream, idx);
      stream << " ";
      WriteStream<int>(stream, output_shape[idx]);
      return;
    }
  }

  // In case the resize is a noop
  WriteStream<int>(stream, 0);
  stream << " ";
  WriteStream<int>(stream, output_shape[0]);
}

template <>
TResizeDimCEmbrio ReadStreamWithoutTypeNamePrefix<TResizeDimCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  auto new_size = ReadStream<int>(stream);
  auto output_shape = shape;
  output_shape[dim] = new_size;
  return {shape, output_shape};
}

template <>
void WriteStream<TBootstrapCEmbrio>(std::ostream& stream,
                                    const TBootstrapCEmbrio& node) {
  stream << TBootstrapCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<Level>(stream, node.UsableLevels());
}

template <>
TBootstrapCEmbrio ReadStreamWithoutTypeNamePrefix<TBootstrapCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  Level usable_levels = ReadStream<Level>(stream);
  return {shape, usable_levels};
}

template <>
void WriteStream<TStrideCEmbrio>(std::ostream& stream,
                                 const TStrideCEmbrio& node) {
  stream << TStrideCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  // Check that we are striding at most one dimension
  CHECK(1 >= Estd::count_if(node.Strides(), [](const Stride& stride) {
          return stride.value() != 1;
        }));
  for (int idx : Estd::indices(node.Strides().size())) {
    if (node.Strides()[idx].value() != 1) {
      WriteStream<int>(stream, idx);
      stream << " ";
      WriteStream<Stride>(stream, node.Strides()[idx]);
      return;
    }
  }
  // In case stride is a noop
  WriteStream<std::string>(stream, "0 1");
}

template <>
TStrideCEmbrio ReadStreamWithoutTypeNamePrefix<TStrideCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  auto stride = ReadStream<int>(stream);
  std::vector<Stride> strides(shape.DimensionCount(), 1);
  strides[dim] = stride;
  return {shape, strides};
}

template <>
void WriteStream<TMergedStrideCEmbrio>(std::ostream& stream,
                                       const TMergedStrideCEmbrio& node) {
  stream << TMergedStrideCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream(stream, node.Strides());
}

template <>
TMergedStrideCEmbrio ReadStreamWithoutTypeNamePrefix<TMergedStrideCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto strides = ReadStream<std::vector<Stride>>(stream);
  return {shape, strides};
}

template <>
void WriteStream<TUnpaddedShiftCEmbrio>(std::ostream& stream,
                                        const TUnpaddedShiftCEmbrio& node) {
  stream << TUnpaddedShiftCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  // Check that shift is done at most along one dimension
  CHECK(1 >= Estd::count_if(node.GetDiffTensorIndex().DimensionIndices(),
                            [](auto& idx) { return idx != 0; }));
  for (int idx :
       Estd::indices(node.GetDiffTensorIndex().DimensionIndices().size())) {
    if (node.GetDiffTensorIndex().DimensionIndices()[idx] != 0) {
      WriteStream<int>(stream, idx);
      stream << " ";
      WriteStream<int>(stream,
                       node.GetDiffTensorIndex().DimensionIndices()[idx]);
      return;
    }
  }
  // In case the shift is a noop
  WriteStream<std::string>(stream, "0 0");
}

template <>
TUnpaddedShiftCEmbrio ReadStreamWithoutTypeNamePrefix<TUnpaddedShiftCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  auto rotate_by = ReadStream<int>(stream);
  std::vector<int> rot_vec(shape.DimensionCount());
  CHECK(dim < shape.DimensionCount());
  CHECK(dim >= 0);
  rot_vec[dim] = rotate_by;
  return {shape, DiffTensorIndex(shape, Array(rot_vec))};
}

template <>
void WriteStream<TCyclicShiftCEmbrio>(std::ostream& stream,
                                      const TCyclicShiftCEmbrio& node) {
  stream << TCyclicShiftCEmbrio::StaticTypeName();
  stream << " ";
  WriteStream<Shape>(stream, node.InputShape());
  // Check that shift is done at most along one dimension
  CHECK(1 >= Estd::count_if(node.GetDiffTensorIndex().DimensionIndices(),
                            [](auto& idx) { return idx != 0; }));
  for (int idx :
       Estd::indices(node.GetDiffTensorIndex().DimensionIndices().size())) {
    if (node.GetDiffTensorIndex().DimensionIndices()[idx] != 0) {
      WriteStream<int>(stream, idx);
      stream << " ";
      WriteStream<int>(stream,
                       node.GetDiffTensorIndex().DimensionIndices()[idx]);
      return;
    }
  }
  // In case the shift is a noop
  WriteStream<std::string>(stream, "0 0");
}

template <>
TCyclicShiftCEmbrio ReadStreamWithoutTypeNamePrefix<TCyclicShiftCEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  auto dim = ReadStream<int>(stream);
  auto rotate_by = ReadStream<int>(stream);
  std::vector<int> rot_vec(shape.DimensionCount());
  CHECK(dim < shape.DimensionCount());
  CHECK(dim >= 0);
  rot_vec[dim] = rotate_by;
  return {shape, DiffTensorIndex(shape, Array(rot_vec))};
}

template <>
void WriteStream<TMulCSIEmbrio>(std::ostream& stream,
                                const TMulCSIEmbrio& node) {
  stream << TMulCSIEmbrio::StaticTypeName() << " ";
  WriteStream<ScaledPtVal>(stream, node.Scalar());
}

template <>
TMulCSIEmbrio ReadStreamWithoutTypeNamePrefix<TMulCSIEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  ScaledPtVal scalar = ReadStream<ScaledPtVal>(stream);
  return {shape, scalar};
}

template <>
void WriteStream<TAddCSIEmbrio>(std::ostream& stream,
                                const TAddCSIEmbrio& node) {
  stream << TAddCSIEmbrio::StaticTypeName() << " ";
  WriteStream<Shape>(stream, node.InputShape());
  stream << " ";
  WriteStream<ScaledPtVal>(stream, node.Scalar());
}

template <>
TAddCSIEmbrio ReadStreamWithoutTypeNamePrefix<TAddCSIEmbrio>(
    std::istream& stream) {
  auto shape = ReadStream<Shape>(stream);
  ScaledPtVal scalar = ReadStream<ScaledPtVal>(stream);
  return {shape, scalar};
}

std::unique_ptr<TOpEmbrio> TOpEmbrio::CreateInstance(std::istream& stream) {
  std::string token = ReadStream<std::string>(stream);
  DerivedRecordType::iterator it = GetMap().find(token);
  if (it == GetMap().end()) {
    LOG(FATAL) << "Unrecognized token " << token;
  }
  return it->second(stream);
}

}  // namespace fhelipe
