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

#ifndef T_OP_EMBRIO_H_
#define T_OP_EMBRIO_H_

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <type_traits>

#include "constants.h"
#include "io_utils.h"
#include "log_scale.h"
#include "scaled_pt_val.h"
#include "t_add_cc.h"
#include "t_add_cp.h"
#include "t_add_csi.h"
#include "t_bootstrap_c.h"
#include "t_chet_repack_c.h"
#include "t_cyclic_shift_c.h"
#include "t_drop_dim_c.h"
#include "t_input_c.h"
#include "t_insert_dim_c.h"
#include "t_mul_cc.h"
#include "t_mul_cp.h"
#include "t_mul_csi.h"
#include "t_output_c.h"
#include "t_reduce_dim_c.h"
#include "t_reorder_dims_c.h"
#include "t_replicate_dim_c.h"
#include "t_resize_dim_c.h"
#include "t_rotate_c.h"
#include "t_stride_c.h"
#include "t_unpadded_shift_c.h"
#include "tensor_index.h"
#include "tensor_layout.h"

namespace fhelipe {

class TOpEmbrio;

template <class T>
std::enable_if_t<std::is_base_of_v<TOpEmbrio, T>, T>
ReadStreamWithoutTypeNamePrefix(std::istream& stream) = delete;

class TOpEmbrio {
 public:
  typedef std::unordered_map<std::string,
                             std::unique_ptr<TOpEmbrio> (*)(std::istream&)>
      DerivedRecordType;
  TOpEmbrio() = default;
  TOpEmbrio(TOpEmbrio&&) = default;
  virtual ~TOpEmbrio() {}
  TOpEmbrio(const TOpEmbrio&) = delete;
  TOpEmbrio& operator=(const TOpEmbrio&) = delete;
  TOpEmbrio& operator=(TOpEmbrio&&) = default;
  // nsamar: The only purpose of the WriteStreamHelper() is to rely
  // on polymorphism to call the correct WriteStream<DerivedT>() function
  virtual void WriteStreamHelper(std::ostream& stream) const = 0;
  virtual const std::string& TypeName() const = 0;
  virtual std::unique_ptr<TOpEmbrio> CloneUniq() const = 0;

  virtual const Shape& OutputShape() const = 0;
  virtual const Shape& InputShape() const = 0;
  virtual std::unique_ptr<TOp> GetTOp(
      const TensorLayout& input_layout,
      const TensorLayout& output_layout) const = 0;
  static std::unique_ptr<TOpEmbrio> CreateInstance(std::istream& stream);

 protected:
  static DerivedRecordType record_map_;
  static DerivedRecordType& GetMap() { return record_map_; }
};

template <class T>
inline std::enable_if_t<std::is_base_of_v<TOpEmbrio, T>, T> ReadStream(
    std::istream& stream) {
  auto token = ReadStream<std::string>(stream);
  CHECK(token == T::StaticTypeName());
  return ReadStreamWithoutTypeNamePrefix<T>(stream);
}

template <>
inline void WriteStream<TOpEmbrio>(std::ostream& stream,
                                   const TOpEmbrio& node) {
  node.WriteStreamHelper(stream);
}

inline TOpEmbrio::DerivedRecordType TOpEmbrio::record_map_ =
    TOpEmbrio::DerivedRecordType{};

// https://stackoverflow.com/questions/582331/is-there-a-way-to-instantiate-objects-from-a-string-holding-their-class-name
template <class T>
class DerivedRegistrar : TOpEmbrio {
 public:
  explicit DerivedRegistrar(const std::string& type_name);
  const Shape& OutputShape() const override { LOG(FATAL); }
  const Shape& InputShape() const override { LOG(FATAL); }
  std::unique_ptr<TOpEmbrio> CloneUniq() const override { LOG(FATAL); }
  std::unique_ptr<TOp> GetTOp(
      const TensorLayout& input_layout,
      const TensorLayout& output_layout) const override {
    (void)input_layout;
    (void)output_layout;
    LOG(FATAL);
  }
  void WriteStreamHelper(std::ostream& stream) const override {
    (void)stream;
    LOG(FATAL);
  }
  const std::string& TypeName() const final { LOG(FATAL); }

 private:
  std::string type_name_;
};

class TAddCCEmbrio;

template <>
void WriteStream<TAddCCEmbrio>(std::ostream& stream, const TAddCCEmbrio& node);

class TAddCCEmbrio final : public TOpEmbrio {
 public:
  explicit TAddCCEmbrio(const Shape& shape) : shape_(shape) {}
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TAddCCEmbrio>(shape_);
  }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TAddCC>(input_layout);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TAddCCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslAddCC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;

  static DerivedRegistrar<TAddCCEmbrio> reg_;
};

class TChetRepackCEmbrio;

template <>
void WriteStream<TChetRepackCEmbrio>(std::ostream& stream,
                                     const TChetRepackCEmbrio& node);

class TChetRepackCEmbrio final : public TOpEmbrio {
 public:
  explicit TChetRepackCEmbrio(const Shape& shape) : shape_(shape) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TChetRepackCEmbrio>(shape_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    auto expected_output_layout = ChetRepackedLayout(
        LogChunkSize(input_layout.ChunkSize()), input_layout.GetShape());
    CHECK(output_layout == expected_output_layout);
    return std::make_unique<TChetRepackC>(input_layout);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TChetRepackCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslChetRepackC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;

  static DerivedRegistrar<TChetRepackCEmbrio> reg_;
};

template <>
TAddCCEmbrio ReadStreamWithoutTypeNamePrefix<TAddCCEmbrio>(
    std::istream& stream);

class TAddCPEmbrio;

template <>
void WriteStream<TAddCPEmbrio>(std::ostream& stream, const TAddCPEmbrio& node);

class TAddCPEmbrio final : public TOpEmbrio {
 public:
  TAddCPEmbrio(const Shape& shape, const std::string& pt_tensor_name,
               const LogScale& pt_tensor_log_scale)
      : shape_(shape),
        pt_tensor_name_(pt_tensor_name),
        pt_tensor_log_scale_(pt_tensor_log_scale) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TAddCPEmbrio>(shape_, pt_tensor_name_,
                                          pt_tensor_log_scale_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TAddCP>(input_layout, pt_tensor_name_,
                                    pt_tensor_log_scale_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TAddCPEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslAddCP; }
  const std::string& TypeName() const final { return StaticTypeName(); }
  const std::string& PtTensorName() const { return pt_tensor_name_; }
  LogScale PtTensorLogScale() const { return pt_tensor_log_scale_; }

 private:
  Shape shape_;
  std::string pt_tensor_name_;
  LogScale pt_tensor_log_scale_;

  static DerivedRegistrar<TAddCPEmbrio> reg_;
};

template <>
TAddCPEmbrio ReadStreamWithoutTypeNamePrefix<TAddCPEmbrio>(
    std::istream& stream);

class TMulCCEmbrio;

template <>
void WriteStream<TMulCCEmbrio>(std::ostream& stream, const TMulCCEmbrio& node);

class TMulCCEmbrio final : public TOpEmbrio {
 public:
  explicit TMulCCEmbrio(const Shape& shape) : shape_(shape) {}
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TMulCCEmbrio>(shape_);
  }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TMulCC>(input_layout);
  }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMulCCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslMulCC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;

  static DerivedRegistrar<TMulCCEmbrio> reg_;
};

template <>
TMulCCEmbrio ReadStreamWithoutTypeNamePrefix<TMulCCEmbrio>(
    std::istream& stream);

class TMulCPEmbrio;

template <>
void WriteStream<TMulCPEmbrio>(std::ostream& stream, const TMulCPEmbrio& node);

class TMulCPEmbrio final : public TOpEmbrio {
 public:
  TMulCPEmbrio(const Shape& shape, const std::string& pt_tensor_name,
               LogScale pt_tensor_log_scale)
      : shape_(shape),
        pt_tensor_name_(pt_tensor_name),
        pt_tensor_log_scale_(pt_tensor_log_scale) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TMulCPEmbrio>(shape_, pt_tensor_name_,
                                          pt_tensor_log_scale_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TMulCP>(input_layout, pt_tensor_name_,
                                    pt_tensor_log_scale_);
  }
  const std::string& PtTensorName() const { return pt_tensor_name_; }
  LogScale PtTensorLogScale() const { return pt_tensor_log_scale_; }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMulCPEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslMulCP; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  std::string pt_tensor_name_;
  LogScale pt_tensor_log_scale_;

  static DerivedRegistrar<TMulCPEmbrio> reg_;
};

template <>
TMulCPEmbrio ReadStreamWithoutTypeNamePrefix<TMulCPEmbrio>(
    std::istream& stream);

class TInputCEmbrio;

template <>
void WriteStream(std::ostream& stream, const TInputCEmbrio& node);

class TInputCEmbrio final : public TOpEmbrio {
 public:
  TInputCEmbrio(const Shape& shape, const std::string& name, LogScale log_scale)
      : shape_(shape), name_(name), log_scale_(log_scale) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TInputCEmbrio>(shape_, name_, log_scale_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TInputC>(input_layout, name_, log_scale_);
  }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TInputCEmbrio>(stream, *this);
  }
  const std::string& TensorName() const { return name_; }
  static const std::string& StaticTypeName() { return kDslInputC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

  LogScale GetLogScale() const { return log_scale_; }

 private:
  Shape shape_;
  std::string name_;
  LogScale log_scale_;

  static DerivedRegistrar<TInputCEmbrio> reg_;
};

template <>
TInputCEmbrio ReadStreamWithoutTypeNamePrefix<TInputCEmbrio>(
    std::istream& stream);

class TOutputCEmbrio;

template <>
void WriteStream<TOutputCEmbrio>(std::ostream& stream,
                                 const TOutputCEmbrio& node);

class TOutputCEmbrio final : public TOpEmbrio {
 public:
  TOutputCEmbrio(const Shape& shape, const std::string& name)
      : shape_(shape), name_(name) {}
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TOutputCEmbrio>(shape_, name_);
  }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TOutputC>(input_layout, name_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TOutputCEmbrio>(stream, *this);
  }

  const std::string& TensorName() const { return name_; }
  static const std::string& StaticTypeName() { return kDslOutputC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  std::string name_;

  static DerivedRegistrar<TOutputCEmbrio> reg_;
};

template <>
TOutputCEmbrio ReadStreamWithoutTypeNamePrefix<TOutputCEmbrio>(
    std::istream& stream);

class TReduceDimCEmbrio;

template <>
void WriteStream<TReduceDimCEmbrio>(std::ostream& stream,
                                    const TReduceDimCEmbrio& node);

class TReduceDimCEmbrio final : public TOpEmbrio {
 public:
  TReduceDimCEmbrio(const Shape& shape, int dimension)
      : shape_(shape),
        output_shape_(GetOutputShapeTReduceDimC(shape, dimension)),
        dimension_(dimension) {}
  const Shape& OutputShape() const final { return output_shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TReduceDimCEmbrio>(shape_, dimension_);
  }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    return std::make_unique<TReduceDimC>(input_layout, output_layout,
                                         dimension_);
  }
  int DimensionToReduce() const { return dimension_; }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TReduceDimCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslReduceDimC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Shape output_shape_;
  int dimension_;

  static DerivedRegistrar<TReduceDimCEmbrio> reg_;
};

template <>
TReduceDimCEmbrio ReadStreamWithoutTypeNamePrefix<TReduceDimCEmbrio>(
    std::istream& stream);

class TReorderDimsCEmbrio;

template <>
void WriteStream<TReorderDimsCEmbrio>(std::ostream& stream,
                                      const TReorderDimsCEmbrio& node);

class TReorderDimsCEmbrio final : public TOpEmbrio {
 public:
  TReorderDimsCEmbrio(const Shape& shape, const std::vector<int>& dim_order)
      : shape_(shape),
        output_shape_(GetOutputShapeTReorderDimsC(shape, dim_order)),
        dim_order_(dim_order) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TReorderDimsCEmbrio>(shape_, dim_order_);
  }
  const Shape& OutputShape() const final { return output_shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    return std::make_unique<TReorderDimsC>(input_layout, output_layout,
                                           dim_order_);
  }
  const std::vector<int>& DimensionOrder() const { return dim_order_; }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TReorderDimsCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslReorderDimsC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Shape output_shape_;
  std::vector<int> dim_order_;

  static DerivedRegistrar<TReorderDimsCEmbrio> reg_;
};

template <>
TReorderDimsCEmbrio ReadStreamWithoutTypeNamePrefix<TReorderDimsCEmbrio>(
    std::istream& stream);

class TReplicateDimCEmbrio;

template <>
void WriteStream<TReplicateDimCEmbrio>(std::ostream& stream,
                                       const TReplicateDimCEmbrio& node);

class TReplicateDimCEmbrio final : public TOpEmbrio {
 public:
  TReplicateDimCEmbrio(const Shape& shape, int dimension, int multiple)
      : shape_(shape),
        output_shape_(
            GetOutputShapeTReplicateDimsC(shape, dimension, multiple)),
        dimension_(dimension),
        multiple_(multiple) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TReplicateDimCEmbrio>(shape_, dimension_,
                                                  multiple_);
  }
  const Shape& OutputShape() const final { return output_shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    return std::make_unique<TReplicateDimC>(input_layout, output_layout,
                                            dimension_, multiple_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TReplicateDimCEmbrio>(stream, *this);
  }
  int DimensionToReplicate() const { return dimension_; }
  int ReplicationMultiple() const { return multiple_; }
  static const std::string& StaticTypeName() { return kDslReplicateDimC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Shape output_shape_;
  int dimension_;
  int multiple_;

  static DerivedRegistrar<TReplicateDimCEmbrio> reg_;
};

template <>
TReplicateDimCEmbrio ReadStreamWithoutTypeNamePrefix<TReplicateDimCEmbrio>(
    std::istream& stream);

class TDropDimCEmbrio;

template <>
void WriteStream<TDropDimCEmbrio>(std::ostream& stream,
                                  const TDropDimCEmbrio& node);

class TDropDimCEmbrio final : public TOpEmbrio {
 public:
  TDropDimCEmbrio(const Shape& shape, int dimension)
      : shape_(shape),
        output_shape_(GetOutputShapeTDropDimC(shape_, dimension)),
        dimension_(dimension) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TDropDimCEmbrio>(shape_, dimension_);
  }
  const Shape& OutputShape() const final { return output_shape_; }
  const Shape& InputShape() const final { return shape_; }
  int DimensionToDrop() const { return dimension_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TDropDimC>(input_layout, dimension_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TDropDimCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslDropDimC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Shape output_shape_;
  int dimension_;

  static DerivedRegistrar<TDropDimCEmbrio> reg_;
};

template <>
TDropDimCEmbrio ReadStreamWithoutTypeNamePrefix<TDropDimCEmbrio>(
    std::istream& stream);

class TInsertDimCEmbrio;

template <>
void WriteStream<TInsertDimCEmbrio>(std::ostream& stream,
                                    const TInsertDimCEmbrio& node);

class TInsertDimCEmbrio final : public TOpEmbrio {
 public:
  TInsertDimCEmbrio(const Shape& shape, int dimension)
      : shape_(shape),
        output_shape_(GetOutputShapeTInsertDimC(shape, dimension)),
        dimension_(dimension) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TInsertDimCEmbrio>(shape_, dimension_);
  }
  const Shape& OutputShape() const final { return output_shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TInsertDimC>(input_layout, dimension_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TInsertDimCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslInsertDimC; }
  const std::string& TypeName() const final { return StaticTypeName(); }
  int DimensionToInsert() const { return dimension_; }

 private:
  Shape shape_;
  Shape output_shape_;
  int dimension_;

  static DerivedRegistrar<TInsertDimCEmbrio> reg_;
};

template <>
TInsertDimCEmbrio ReadStreamWithoutTypeNamePrefix<TInsertDimCEmbrio>(
    std::istream& stream);

class TResizeDimCEmbrio;

template <>
void WriteStream<TResizeDimCEmbrio>(std::ostream& stream,
                                    const TResizeDimCEmbrio& node);

class TResizeDimCEmbrio final : public TOpEmbrio {
 public:
  TResizeDimCEmbrio(const Shape& shape, const Shape& output_shape)
      : shape_(shape), output_shape_(output_shape) {}
  const Shape& OutputShape() const final { return output_shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TResizeDimCEmbrio>(shape_, output_shape_);
  }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(output_shape_ == output_layout.GetShape());
    return std::make_unique<TResizeDimC>(input_layout, output_layout);
  }

  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TResizeDimCEmbrio>(stream, *this);
  }

  static const std::string& StaticTypeName() { return kDslResizeDimC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Shape output_shape_;

  static DerivedRegistrar<TResizeDimCEmbrio> reg_;
};

template <>
TResizeDimCEmbrio ReadStreamWithoutTypeNamePrefix<TResizeDimCEmbrio>(
    std::istream& stream);

class TStrideCEmbrio;

template <>
void WriteStream<TStrideCEmbrio>(std::ostream& stream,
                                 const TStrideCEmbrio& node);

class TStrideCEmbrio final : public TOpEmbrio {
 public:
  TStrideCEmbrio(const Shape& shape, const std::vector<Stride>& strides)
      : shape_(shape),
        output_shape_(GetOutputShapeTStrideC(shape, strides)),
        strides_(strides) {}
  const Shape& OutputShape() const final { return output_shape_; }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TStrideCEmbrio>(shape_, strides_);
  }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    return std::make_unique<TStrideC>(input_layout, output_layout, strides_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TStrideCEmbrio>(stream, *this);
  }
  const std::vector<Stride>& Strides() const { return strides_; }
  static const std::string& StaticTypeName() { return kDslStrideDimC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Shape output_shape_;
  std::vector<Stride> strides_;

  static DerivedRegistrar<TStrideCEmbrio> reg_;
};

template <>
TStrideCEmbrio ReadStreamWithoutTypeNamePrefix<TStrideCEmbrio>(
    std::istream& stream);

class TMergedStrideCEmbrio;

template <>
void WriteStream<TMergedStrideCEmbrio>(std::ostream& stream,
                                       const TMergedStrideCEmbrio& node);

class TMergedStrideCEmbrio final : public TOpEmbrio {
 public:
  TMergedStrideCEmbrio(const Shape& shape, const std::vector<Stride>& strides)
      : shape_(shape),
        output_shape_(GetOutputShapeTStrideC(shape, strides)),
        strides_(strides) {}
  const Shape& OutputShape() const final { return output_shape_; }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TMergedStrideCEmbrio>(shape_, strides_);
  }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    return std::make_unique<TStrideC>(input_layout, output_layout, strides_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMergedStrideCEmbrio>(stream, *this);
  }
  const std::vector<Stride>& Strides() const { return strides_; }
  static const std::string& StaticTypeName() { return kDslMergedStrideDimC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Shape output_shape_;
  std::vector<Stride> strides_;

  static DerivedRegistrar<TMergedStrideCEmbrio> reg_;
};

template <>
TMergedStrideCEmbrio ReadStreamWithoutTypeNamePrefix<TMergedStrideCEmbrio>(
    std::istream& stream);

class TUnpaddedShiftCEmbrio;

template <>
void WriteStream<TUnpaddedShiftCEmbrio>(std::ostream& stream,
                                        const TUnpaddedShiftCEmbrio& node);

class TUnpaddedShiftCEmbrio final : public TOpEmbrio {
 public:
  TUnpaddedShiftCEmbrio(const Shape& shape, const DiffTensorIndex& rotate_by)
      : shape_(shape), rotate_by_(rotate_by) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TUnpaddedShiftCEmbrio>(shape_, rotate_by_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TUnpaddedShiftC>(input_layout, rotate_by_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TUnpaddedShiftCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslUnpaddedShiftC; }
  const std::string& TypeName() const final { return StaticTypeName(); }
  const DiffTensorIndex& GetDiffTensorIndex() const { return rotate_by_; }

 private:
  Shape shape_;
  DiffTensorIndex rotate_by_;

  static DerivedRegistrar<TUnpaddedShiftCEmbrio> reg_;
};

class TCyclicShiftCEmbrio;

template <>
void WriteStream<TCyclicShiftCEmbrio>(std::ostream& stream,
                                      const TCyclicShiftCEmbrio& node);

class TCyclicShiftCEmbrio final : public TOpEmbrio {
 public:
  TCyclicShiftCEmbrio(const Shape& shape, const DiffTensorIndex& rotate_by)
      : shape_(shape), rotate_by_(rotate_by) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TCyclicShiftCEmbrio>(shape_, rotate_by_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TCyclicShiftC>(input_layout, rotate_by_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TCyclicShiftCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslCyclicShiftC; }
  const std::string& TypeName() const final { return StaticTypeName(); }
  const DiffTensorIndex& GetDiffTensorIndex() const { return rotate_by_; }

 private:
  Shape shape_;
  DiffTensorIndex rotate_by_;

  static DerivedRegistrar<TCyclicShiftCEmbrio> reg_;
};

class TRotateCEmbrio;

template <>
void WriteStream<TRotateCEmbrio>(std::ostream& stream,
                                 const TRotateCEmbrio& node);

class TRotateCEmbrio final : public TOpEmbrio {
 public:
  TRotateCEmbrio(const Shape& shape, const int& rotate_by)
      : shape_(shape), rotate_by_(rotate_by) {
    CHECK(shape_.DimensionCount() == 1);
  }
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TRotateCEmbrio>(shape_, rotate_by_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TRotateC>(input_layout, rotate_by_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TRotateCEmbrio>(stream, *this);
  }
  static const std::string& StaticTypeName() { return kDslRotateC; }
  const std::string& TypeName() const final { return StaticTypeName(); }
  int RotateBy() const { return rotate_by_; }

 private:
  Shape shape_;
  int rotate_by_;

  static DerivedRegistrar<TRotateCEmbrio> reg_;
};

class TBootstrapCEmbrio;

template <>
void WriteStream<TBootstrapCEmbrio>(std::ostream& stream,
                                    const TBootstrapCEmbrio& node);

class TBootstrapCEmbrio final : public TOpEmbrio {
 public:
  TBootstrapCEmbrio(const Shape& shape, Level usable_levels)
      : shape_(shape), usable_levels_(usable_levels) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TBootstrapCEmbrio>(shape_, usable_levels_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TBootstrapC>(input_layout, usable_levels_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TBootstrapCEmbrio>(stream, *this);
  }
  Level UsableLevels() const { return usable_levels_; }
  static const std::string& StaticTypeName() { return kDslBootstrapC; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  Level usable_levels_;

  static DerivedRegistrar<TBootstrapCEmbrio> reg_;
};

template <>
TBootstrapCEmbrio ReadStreamWithoutTypeNamePrefix<TBootstrapCEmbrio>(
    std::istream& stream);

template <>
TUnpaddedShiftCEmbrio ReadStreamWithoutTypeNamePrefix<TUnpaddedShiftCEmbrio>(
    std::istream& stream);

class TMulCSIEmbrio;

template <>
void WriteStream<TMulCSIEmbrio>(std::ostream& stream,
                                const TMulCSIEmbrio& node);

class TMulCSIEmbrio final : public TOpEmbrio {
 public:
  TMulCSIEmbrio(const Shape& shape, const ScaledPtVal& scalar)
      : shape_(shape), scalar_(scalar) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TMulCSIEmbrio>(shape_, scalar_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TMulCSI>(input_layout, scalar_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TMulCSIEmbrio>(stream, *this);
  }
  const ScaledPtVal& Scalar() const { return scalar_; }
  static const std::string& StaticTypeName() { return kDslMulCSI; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  ScaledPtVal scalar_;

  static DerivedRegistrar<TMulCSIEmbrio> reg_;
};

class TAddCSIEmbrio;

template <>
void WriteStream<TAddCSIEmbrio>(std::ostream& stream,
                                const TAddCSIEmbrio& node);

class TAddCSIEmbrio final : public TOpEmbrio {
 public:
  TAddCSIEmbrio(const Shape& shape, const ScaledPtVal& scalar)
      : shape_(shape), scalar_(scalar) {}
  std::unique_ptr<TOpEmbrio> CloneUniq() const final {
    return std::make_unique<TAddCSIEmbrio>(shape_, scalar_);
  }
  const Shape& OutputShape() const final { return shape_; }
  const Shape& InputShape() const final { return shape_; }
  std::unique_ptr<TOp> GetTOp(const TensorLayout& input_layout,
                              const TensorLayout& output_layout) const final {
    CHECK(shape_ == input_layout.GetShape());
    CHECK(input_layout == output_layout);
    return std::make_unique<TAddCSI>(input_layout, scalar_);
  }
  void WriteStreamHelper(std::ostream& stream) const final {
    WriteStream<TAddCSIEmbrio>(stream, *this);
  }
  const ScaledPtVal& Scalar() const { return scalar_; }
  static const std::string& StaticTypeName() { return kDslAddCSI; }
  const std::string& TypeName() const final { return StaticTypeName(); }

 private:
  Shape shape_;
  ScaledPtVal scalar_;

  static DerivedRegistrar<TAddCSIEmbrio> reg_;
};

template <>
TMulCSIEmbrio ReadStreamWithoutTypeNamePrefix<TMulCSIEmbrio>(
    std::istream& stream);

template <>
TAddCSIEmbrio ReadStreamWithoutTypeNamePrefix<TAddCSIEmbrio>(
    std::istream& stream);

template <class T>
DerivedRegistrar<T>::DerivedRegistrar(const std::string& type_name)
    : type_name_(type_name) {
  GetMap().emplace(std::string(type_name), [](std::istream& stream) {
    return std::unique_ptr<TOpEmbrio>{
        new T{std::move(ReadStreamWithoutTypeNamePrefix<T>(stream))}};
  });
}

}  // namespace fhelipe

#endif  // T_OP_EMBRIO_H_
