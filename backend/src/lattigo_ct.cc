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

#include "include/lattigo_ct.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "include/checker.h"
#include "include/cleartext.h"
#include "include/environment.h"
#include "include/filesystem_utils.h"
#include "include/index_mask.h"
#include "include/io_utils.h"
#include "include/level.h"
#include "include/level_info.h"
#include "include/log_scale.h"
#include "include/plaintext.h"
#include "include/plaintext_chunk.h"
#include "include/program_context.h"
#include "latticpp/ckks/bootstrap.h"
#include "latticpp/ckks/bootstrap_params.h"
#include "latticpp/ckks/ciphertext.h"
#include "latticpp/ckks/decryptor.h"
#include "latticpp/ckks/encoder.h"
#include "latticpp/ckks/encryptor.h"
#include "latticpp/ckks/evaluator.h"
#include "latticpp/ckks/keygen.h"
#include "latticpp/ckks/lattigo_param.h"
#include "latticpp/ckks/marshaler.h"
#include "latticpp/ckks/params.h"
#include "utils.h"

using namespace latticpp;

namespace fhelipe {

static std::string marshal_public_key = "public_key.marshal";
static std::string marshal_secret_key = "secret_key.marshal";

namespace {

std::filesystem::path MarshalFolder(const ProgramContext& context) {
  static const std::filesystem::path marshal_folder_base =
      "/tmp/lattigo_" + env::Username();
  return marshal_folder_base /
         context.GetLattigoParam().GetMarshalFolderExtension();
}

}  // namespace

LogScale LattigoCt::GetLogScale() const { return GetLevelInfo().LogScale(); }

Level LattigoCt::GetLevel() const { return GetLevelInfo().Level(); }

double LogScaleToScale(LogScale log_scale) {
  return std::pow(2.0, log_scale.value());
}

LattigoCt LattigoCt::ZeroC(const ProgramContext& context,
                           const LevelInfo& level_info) {
  if (!scheme_initialized_) {
    InitScheme(context);
  }
  auto pt_chunk =
      PtChunk(std::vector<PtVal>(1 << context.GetLogChunkSize().value()));
  Plaintext pt = encodeNTTAtLvlNew(*params_, *encoder_, pt_chunk.Values(),
                                   level_info.Level().value(),
                                   LogScaleToScale(level_info.LogScale()));
  return LattigoCt{encryptNew(context.GetLattigoParam(), *encryptor_, pt),
                   Cleartext::ZeroC(context, level_info)};
}

LattigoCt LattigoCt::Encrypt(const ProgramContext& context,
                             const PtChunk& pt_chunk) {
  if (!scheme_initialized_) {
    InitScheme(context);
  }
  Plaintext pt = encodeNTTAtLvlNew(*params_, *encoder_, pt_chunk.Values(),
                                   context.UsableLevels().value(),
                                   LogScaleToScale(context.LogScale()));
  return LattigoCt{
      encryptNew(context.GetLattigoParam(), *encryptor_, pt),
      Cleartext(pt_chunk, {context.UsableLevels(), context.LogScale()})};
}

LattigoCt::LattigoCt(const latticpp::Ciphertext& ciphertext,
                     const Cleartext& cleartext)
    : ciphertext_(ciphertext), cleartext_(cleartext) {
  if (!scheme_initialized_) {
    InitScheme(MakeProgramContext(ciphertext.GetLattigoParam()));
  }
}

LattigoCt::LattigoCt(const latticpp::Ciphertext& ciphertext)
    : ciphertext_(ciphertext), cleartext_(PtChunk{{0}}, LevelInfo{5, 50}) {
  if (!scheme_initialized_) {
    InitScheme(MakeProgramContext(ciphertext.GetLattigoParam()));
  }

  cleartext_ = {PtChunk(decode(*encoder_, decryptNew(*decryptor_, ciphertext_),
                               logSlots(*params_))),
                GetLevelInfo()};
}

LattigoCt LattigoCt::MulCC(const LattigoCt& rhs) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));
  auto result = copyNew(ciphertext_);
  result = mulRelinNew(*evaluator_, result, rhs.GetCt());
  return LattigoCt(result, GetCleartext().MulCC(rhs.GetCleartext()));
}

LattigoCt LattigoCt::AddCC(const LattigoCt& rhs) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));
  auto tmp = copyNew(ciphertext_);
  auto rhs_ct = copyNew(rhs.GetCt());
  uint64_t target_level = std::min(level(rhs_ct), level(tmp));
  auto target_scale = std::max(scale(rhs_ct), scale(tmp));
  auto result = newCiphertext(ciphertext_.GetLattigoParam(), *params_, 1,
                              target_level, target_scale);
  dropLevel(*evaluator_, rhs_ct, level(rhs_ct) - target_level);
  dropLevel(*evaluator_, tmp, level(tmp) - target_level);
  add(*evaluator_, tmp, rhs_ct, result);
  return LattigoCt(result, GetCleartext().AddCC(rhs.GetCleartext()));
}

LattigoCt LattigoCt::BootstrapC(Level usable_levels) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));
  LOG(INFO) << "Start";
  TestCloseEnough(Decrypt().Values(), GetCleartext().Decrypt().Values());
  CheckSmallEnoughForBootstrapping(GetCleartext().Decrypt().Values());
  if (!bootstrapper_) {
    bootstrapper_ = std::make_unique<::latticpp::Bootstrapper>(
        newBootstrapper(*params_, *boot_params_, *secret_key_));
  }
  auto tmp = copyNew(ciphertext_);
  LOG(INFO) << "Scale before boot: " << scale(tmp);
  setScale(*evaluator_, tmp, scale(*params_));
  Ciphertext result = bootstrap(*bootstrapper_, tmp);
  auto latti_result =
      LattigoCt(result, GetCleartext().BootstrapC(usable_levels));
  LOG(INFO) << "End" << std::endl;
  LOG(INFO) << "Ct: ";
  auto lhs = latti_result.Decrypt().Values();
  auto rhs = GetCleartext().Decrypt().Values();
  WriteStream(LOG(INFO), lhs);
  LOG(INFO) << std::endl << "Clear: ";
  WriteStream(LOG(INFO), rhs);
  LOG(INFO) << std::endl;
  LOG(INFO) << "DIFF: " << LInfinityDiff(lhs, rhs) << std::endl;
  TestCloseEnough(lhs, rhs);
  return latti_result;
}

LattigoCt LattigoCt::AddCP(const ScaledPtChunk& pt_chunk) const {
  auto result = copyNew(ciphertext_);
  Plaintext pt = encodeNTTAtLvlNew(
      *params_, *encoder_, pt_chunk.chunk().Values(), level(ciphertext_),
      LogScaleToScale(pt_chunk.GetLogScale()));
  addPlain(*evaluator_, result, pt, result);
  return LattigoCt(result, GetCleartext().AddCP(pt_chunk));
}

LattigoCt LattigoCt::MulCP(const ScaledPtChunk& pt_chunk) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));
  auto result = copyNew(ciphertext_);
  Plaintext pt = encodeNTTAtLvlNew(
      *params_, *encoder_, pt_chunk.chunk().Values(), level(ciphertext_),
      LogScaleToScale(pt_chunk.GetLogScale()));
  mulPlain(*evaluator_, result, pt, result);
  return LattigoCt(result, GetCleartext().MulCP(pt_chunk));
}

LattigoCt LattigoCt::AddCS(const ScaledPtVal& scalar) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));
  auto result = copyNew(ciphertext_);
  addConst(*evaluator_, result, scalar.value(), result);
  return LattigoCt(result, GetCleartext().AddCS(scalar));
}

LattigoCt LattigoCt::RescaleC(LogScale rescale_amount) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));

  auto result = copyNew(ciphertext_);
  // nsamar: The `- 5` is there to ensure the rescale happens, because
  // lattigo only rescales by full primes
  rescale(*evaluator_, result,
          LogScaleToScale(LogScale(GetLogScale() - rescale_amount - 5)),
          result);
  CHECK(GetLogScale().value() > std::log2(scale(result)))
      << GetLogScale().value() << " " << std::log2(scale(result)) << " "
      << rescale_amount.value();
  auto latti_result =
      LattigoCt(result, GetCleartext().RescaleC(rescale_amount));
  return latti_result;
}

LattigoCt LattigoCt::RotateC(int rotate_by) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));
  int slot_count = 1 << (ciphertext_.GetLattigoParam().LogN() - 1);

  int old_rotate_by = rotate_by;
  if (IsPowerOfTwo(rotate_by)) {
    auto result = copyNew(ciphertext_);
    rotate(*evaluator_, result, -rotate_by, result);
    return LattigoCt(result, GetCleartext().RotateC(old_rotate_by));
  }
  rotate_by = slot_count - rotate_by;
  while (rotate_by < 0) {
    rotate_by += slot_count;
  }
  rotate_by %= slot_count;
  auto result = copyNew(ciphertext_);
  std::vector<int> powers = MaskedIndices(std::abs(rotate_by));
  for (int power : powers) {
    int curr_rotate_by = (1 << power);
    rotate(*evaluator_, result, curr_rotate_by, result);
  }
  auto latti_result = LattigoCt(result, GetCleartext().RotateC(old_rotate_by));
  return latti_result;
}

LattigoCt LattigoCt::MulCS(const ScaledPtVal& scalar) const {
  InitEvaluator(MakeProgramContext(ciphertext_.GetLattigoParam()));
  auto result = copyNew(ciphertext_);
  multByConst(*evaluator_, result, scalar.value(), result);
  return LattigoCt(result, GetCleartext().MulCS(scalar));
}

LevelInfo LattigoCt::GetLevelInfo() const {
  return {static_cast<int>(level(ciphertext_)),
          static_cast<int>(std::log2(std::round(scale(ciphertext_))))};
}

void LattigoCt::InitEvaluator(const ProgramContext& context) {
  if (!scheme_initialized_) {
    InitScheme(context);
  }
  if (evaluator_) {
    return;
  }
  std::vector<int> keys_for_amounts;
  for (int i = 0; i < 15; i++) {
    keys_for_amounts.push_back(1 << i);
    keys_for_amounts.push_back((1 << 15) - (1 << i));
  }
  rotation_keys_ = std::make_unique<RotationKeys>(
      genRotationKeysForRotations(*kgen_, *secret_key_, keys_for_amounts));
  relin_key_ =
      std::make_unique<RelinearizationKey>(genRelinKey(*kgen_, *secret_key_));

  eval_key_ = std::make_unique<EvaluationKey>(
      makeEvaluationKey(*relin_key_, *rotation_keys_));
  evaluator_ = std::make_unique<Evaluator>(newEvaluator(*params_, *eval_key_));
}

void LattigoCt::InitScheme(const ProgramContext& context) {
  CHECK(!scheme_initialized_);
  latticpp::LattigoParam param = context.GetLattigoParam();

  boot_params_ =
      std::make_unique<BootstrappingParameters>(getBootstrappingParams(param));

  params_ = std::make_unique<Parameters>(getDefaultCKKSParams(param));

  kgen_ = std::make_unique<KeyGenerator>(newKeyGenerator(*params_));
  std::filesystem::path marshal_folder = MarshalFolder(context);

  if (!Exists(marshal_folder)) {
    EnsureDirectoryExists(marshal_folder);
    struct KeyPairHandle key_pair = genKeyPair(*kgen_);
    secret_key_ = std::make_unique<SecretKey>(key_pair.sk);
    public_key_ = std::make_unique<PublicKey>(key_pair.pk);
    auto os_secret_key =
        OpenStream<std::ofstream>(marshal_folder / marshal_secret_key);
    marshalBinarySecretKey(*secret_key_, os_secret_key);
    auto os_public_key =
        OpenStream<std::ofstream>(marshal_folder / marshal_public_key);
    marshalBinaryPublicKey(*public_key_, os_public_key);
  } else {
    auto iss_secret_key =
        OpenStream<std::ifstream>(marshal_folder / marshal_secret_key);
    auto iss_public_key =
        OpenStream<std::ifstream>(marshal_folder / marshal_public_key);
    secret_key_ =
        std::make_unique<SecretKey>(unmarshalBinarySecretKey(iss_secret_key));
    public_key_ =
        std::make_unique<PublicKey>(unmarshalBinaryPublicKey(iss_public_key));
  }

  encoder_ = std::make_unique<Encoder>(newEncoder(*params_));
  decryptor_ =
      std::make_unique<Decryptor>(newDecryptor(*params_, *secret_key_));
  encryptor_ =
      std::make_unique<Encryptor>(newEncryptorFromPk(*params_, *public_key_));

  scheme_initialized_ = true;
}

template <>
void WriteStream<LattigoCt>(std::ostream& stream, const LattigoCt& ct) {
  return marshalBinaryCiphertext(ct.GetCt(), stream);
}

template <>
LattigoCt ReadStream<LattigoCt>(std::istream& stream) {
  return LattigoCt{unmarshalBinaryCiphertext(stream)};
}

PtChunk LattigoCt::Decrypt() const {
  auto result = PtChunk(decode(*encoder_, decryptNew(*decryptor_, ciphertext_),
                               logSlots(*params_)));
  return result;
}

}  // namespace fhelipe
