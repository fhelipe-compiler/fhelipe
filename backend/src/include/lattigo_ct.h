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

#ifndef FHELIPE_LATTIGO_CT_H_
#define FHELIPE_LATTIGO_CT_H_

#include <latticpp/latticpp.h>

#include <iosfwd>
#include <memory>

#include "cleartext.h"
#include "constants.h"
#include "include/program_context.h"
#include "latticpp/ckks/lattigo_param.h"
#include "latticpp/marshal/gohandle.h"
#include "level.h"
#include "level_info.h"
#include "plaintext.h"
#include "plaintext_chunk.h"
#include "scaled_pt_chunk.h"
#include "scaled_pt_val.h"

namespace latticpp {
class LattigoParam;
}  // namespace latticpp

namespace fhelipe {

class LattigoCt {
 public:
  explicit LattigoCt(const latticpp::Ciphertext& ciphertext);
  explicit LattigoCt(const latticpp::Ciphertext& ciphertext,
                     const Cleartext& cleartext);

  LattigoCt MulCC(const LattigoCt& rhs) const;
  LattigoCt MulCP(const ScaledPtChunk& cmsg) const;
  LattigoCt MulCS(const ScaledPtVal& scalar) const;

  LattigoCt AddCC(const LattigoCt& rhs) const;
  LattigoCt AddCP(const ScaledPtChunk& cmsg) const;
  LattigoCt AddCS(const ScaledPtVal& scalar) const;

  LattigoCt RotateC(int rotate_by) const;
  LattigoCt RescaleC(LogScale rescale_amount) const;
  ChunkSize GetChunkSize() const {
    // TODO(nsamar): Actually implement
    return 1;
  }

  LattigoCt BootstrapC(Level usable_levels) const;

  LevelInfo GetLevelInfo() const;
  LogScale GetLogScale() const;
  Level GetLevel() const;

  PtChunk Decrypt() const;
  static LattigoCt Encrypt(const ProgramContext& param, const PtChunk& values);
  const latticpp::Ciphertext& GetCt() const { return ciphertext_; }
  static LattigoCt ZeroC(const ProgramContext& context,
                         const LevelInfo& level_info);

  const Cleartext& GetCleartext() const { return cleartext_; }

 private:
  ::latticpp::Ciphertext ciphertext_;
  Cleartext cleartext_;
  static void InitScheme(const ProgramContext& param);
  static void InitEvaluator(const ProgramContext& param);

  static int logp;

  static std::unique_ptr<::latticpp::EvaluationKey> boot_key_;
  static std::unique_ptr<::latticpp::EvaluationKey> eval_key_;
  static std::unique_ptr<::latticpp::Encryptor> encryptor_;
  static std::unique_ptr<::latticpp::BootstrappingParameters> boot_params_;
  static std::unique_ptr<::latticpp::KeyGenerator> kgen_;
  static std::unique_ptr<::latticpp::RelinearizationKey> relin_key_;
  static std::unique_ptr<::latticpp::RotationKeys> rotation_keys_;
  static std::unique_ptr<::latticpp::SecretKey> secret_key_;
  static std::unique_ptr<::latticpp::PublicKey> public_key_;
  static std::unique_ptr<::latticpp::Decryptor> decryptor_;
  static std::unique_ptr<::latticpp::Evaluator> evaluator_;
  static std::unique_ptr<::latticpp::Encoder> encoder_;
  static std::unique_ptr<::latticpp::Parameters> params_;
  static std::unique_ptr<::latticpp::Bootstrapper> bootstrapper_;
  static bool scheme_initialized_;
};

inline std::unique_ptr<::latticpp::SecretKey> LattigoCt::secret_key_ = nullptr;
inline std::unique_ptr<::latticpp::PublicKey> LattigoCt::public_key_ = nullptr;
inline std::unique_ptr<::latticpp::EvaluationKey> LattigoCt::boot_key_ =
    nullptr;
inline std::unique_ptr<::latticpp::EvaluationKey> LattigoCt::eval_key_ =
    nullptr;
inline std::unique_ptr<::latticpp::BootstrappingParameters>
    LattigoCt::boot_params_ = nullptr;
inline std::unique_ptr<::latticpp::KeyGenerator> LattigoCt::kgen_ = nullptr;
inline std::unique_ptr<::latticpp::RelinearizationKey> LattigoCt::relin_key_ =
    nullptr;
inline std::unique_ptr<::latticpp::RotationKeys> LattigoCt::rotation_keys_ =
    nullptr;
inline bool LattigoCt::scheme_initialized_ = false;
inline std::unique_ptr<::latticpp::Encryptor> LattigoCt::encryptor_ = nullptr;
inline std::unique_ptr<::latticpp::Decryptor> LattigoCt::decryptor_ = nullptr;
inline std::unique_ptr<::latticpp::Evaluator> LattigoCt::evaluator_ = nullptr;
inline std::unique_ptr<::latticpp::Encoder> LattigoCt::encoder_ = nullptr;
inline std::unique_ptr<::latticpp::Parameters> LattigoCt::params_ = nullptr;
inline std::unique_ptr<::latticpp::Bootstrapper> LattigoCt::bootstrapper_ =
    nullptr;

template <>
void WriteStream<LattigoCt>(std::ostream& os, const LattigoCt& ct);

template <class T>
T Encrypt(const ProgramContext& context, const PtChunk& input_tensor);

template <>
LattigoCt ReadStream<LattigoCt>(std::istream& iss);

template <class T>
T Encrypt(const PtChunk& input_tensor, const ProgramContext& context);

template <>
inline LattigoCt Encrypt<LattigoCt>(const PtChunk& input_tensor,
                                    const ProgramContext& context) {
  return LattigoCt::Encrypt(context, input_tensor);
}

}  // namespace fhelipe

#endif  // FHELIPE_LATTIGO_CT_H_
