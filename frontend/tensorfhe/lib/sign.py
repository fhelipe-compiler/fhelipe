# $lic$
# Copyright (C) 2023-2024 by Massachusetts Institute of Technology
#
# This file is part of the Fhelipe compiler.
#
# Fhelipe is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# Fhelipe is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

from typing import Callable, Sequence

from ..core import VectorT, _result_repack
from .utils import poly_eval


@_result_repack
def relu_from_sign(x: VectorT, s: VectorT) -> VectorT:
    return 0.5 * (x + x * s)


def relu(x: VectorT, alpha: int = 14) -> VectorT:
    return relu_from_sign(x, sign(x, alpha=alpha))


def maximum(seq: Sequence[VectorT]) -> VectorT:
    """
    Element-wise maximum of tensors

    Requires that elements are in [-1, 1]
    """
    if len(seq) == 0:
        raise ValueError("maximum requires a non-empty sequence", seq)
    elif len(seq) == 1:
        return seq[0]
    else:
        mid = len(seq) // 2
        max_l = maximum(seq[:mid])
        max_r = maximum(seq[mid:])

        diff = max_l + (-1 * max_r)
        # Multiplying by 0.5 to make sure `sign` input is in [-1, 1]
        diff_sign = sign(diff * 0.5)
        diff_abs = diff_sign * diff

        return 0.5 * (max_l + max_r + diff_abs)


def sign_polynomials(alpha: int) -> Sequence[Callable[[VectorT], VectorT]]:
    def poly_f(c: Sequence[float]) -> Callable[[VectorT], VectorT]:
        return lambda x: poly_eval(x, c)

    return tuple(poly_f(c) for c in sign_coeffs[alpha])


def sign(x: VectorT, alpha: int = 14) -> VectorT:
    for p in sign_polynomials(alpha):
        x = p(x)
    return x


sign_coeffs = {
    8: (
        (
            0,
            +8.83133072022416e00,
            0,
            -4.64575039895512e01,
            0,
            +8.30282234720408e01,
            0,
            -4.49928477828070e01,
        ),
        (
            0,
            +3.94881885083263e00,
            0,
            -1.29103010992282e01,
            0,
            +2.80865362174658e01,
            0,
            -3.55969148965137e01,
            0,
            +2.65159370881337e01,
            0,
            -1.14184889368449e01,
            0,
            +2.62558443881334e00,
            0,
            -2.49172299998642e-01,
        ),
    ),
    12: (
        (
            0,
            +1.15523042357223e01,
            0,
            -6.77794513440968e01,
            0,
            +1.25283740404562e02,
            0,
            -6.90142908232934e01,
        ),
        (
            0,
            +9.65167636181626e00,
            0,
            -6.16939174538469e01,
            0,
            +1.55170351652298e02,
            0,
            -1.82697582383214e02,
            0,
            +1.12910726525406e02,
            0,
            -3.77752411770263e01,
            0,
            +6.47503909732344e00,
            0,
            -4.45613365723361e-01,
        ),
        (
            0,
            +5.25888355571745e00,
            0,
            -3.37233593794284e01,
            0,
            +1.64983085013457e02,
            0,
            -5.41408891406992e02,
            0,
            +1.22296207997963e03,
            0,
            -1.95201910566479e03,
            0,
            +2.24084021378300e03,
            0,
            -1.86634916983170e03,
            0,
            +1.12722117843121e03,
            0,
            -4.88070474638380e02,
            0,
            +1.47497846308920e02,
            0,
            -2.95171048879526e01,
            0,
            +3.51269520930994e00,
            0,
            -1.88101836557879e-01,
        ),
    ),
    13: (
        (
            0,  # 00
            +2.455_894_154_250_04e01,  # 01
            0,  # 02
            -6.696_604_497_168_94e02,  # 03
            0,  # 04
            +6.672_998_483_013_39e03,  # 05
            0,  # 06
            -3.060_366_561_638_98e04,  # 07
            0,  # 08
            +7.318_840_329_877_87e04,  # 09
            0,  # 10
            -9.444_332_170_500_84e04,  # 11
            0,  # 12
            +6.232_540_942_125_46e04,  # 13
            0,  # 14
            -1.649_467_441_178_05e04,  # 15
        ),
        (
            0,  # 00
            +9.356_256_360_354_39e00,  # 01
            0,  # 02
            -5.916_389_639_336_26e01,  # 03
            0,  # 04
            +1.488_609_306_264_48e02,  # 05
            0,  # 06
            -1.758_128_748_785_82e02,  # 07
            0,  # 08
            +1.091_112_996_859_55e02,  # 09
            0,  # 10
            -3.667_688_399_787_55e01,  # 11
            0,  # 12
            +6.318_462_903_112_94e00,  # 13
            0,  # 14
            -4.371_134_150_821_77e-01,  # 15
        ),
        (
            0,  # 00
            +5.078_135_697_588_61e00,  # 01
            0,  # 02
            -3.073_299_181_371_86e01,  # 03
            0,  # 04
            +1.441_097_468_128_09e02,  # 05
            0,  # 06
            -4.596_616_888_261_42e02,  # 07
            0,  # 08
            +1.021_520_644_704_59e03,  # 09
            0,  # 10
            -1.620_562_567_088_77e03,  # 11
            0,  # 12
            +1.864_676_464_165_70e03,  # 13
            0,  # 14
            -1.567_493_008_771_43e03,  # 15
            0,  # 16
            +9.609_703_090_934_22e02,  # 17
            0,  # 18
            -4.243_261_618_716_46e02,  # 19
            0,  # 20
            +1.312_785_092_560_03e02,  # 21
            0,  # 22
            -2.698_125_766_261_15e01,  # 23
            0,  # 24
            +3.306_513_873_155_65e00,  # 25
            0,  # 26
            -1.827_429_446_275_33e-01,  # 27
        ),
    ),
    14: (
        (
            0,
            +2.490_521_431_937_540e01,  # 01
            0,
            -6.823_830_575_824_300e02,  # 03
            0,
            +6.809_428_453_905_990e03,  # 05
            0,
            -3.125_071_000_171_050e04,  # 07
            0,
            +7.476_593_883_637_570e04,  # 09
            0,
            -9.650_468_384_758_390e04,  # 11
            0,
            +6.369_779_237_782_460e04,  # 13
            0,
            -1.686_026_213_471_900e04,  # 15
        ),
        (
            0,
            +1.682_855_119_260_110e01,  # 01
            0,
            -3.398_117_504_956_590e02,  # 03
            0,
            +2.790_699_987_938_470e03,  # 05
            0,
            -1.135_141_515_737_900e04,  # 07
            0,
            +2.662_300_102_837_450e04,  # 09
            0,
            -3.938_403_286_619_750e04,  # 11
            0,
            +3.878_842_303_480_600e04,  # 13
            0,
            -2.623_953_038_449_880e04,  # 15
            0,
            +1.236_562_070_165_320e04,  # 17
            0,
            -4.053_364_600_899_990e03,  # 19
            0,
            +9.060_428_809_510_870e02,  # 21
            0,
            -1.316_876_492_082_880e02,  # 23
            0,
            +1.121_760_790_336_230e01,  # 25
            0,
            -4.249_380_204_674_710e-01,  # 27
        ),
        (
            0,
            +5.317_554_976_893_910e00,  # 01
            0,
            -3.543_715_315_315_770e01,  # 03
            0,
            +1.841_224_413_291_400e02,  # 05
            0,
            -6.553_868_301_462_530e02,  # 07
            0,
            +1.638_783_354_280_600e03,  # 09
            0,
            -2.953_862_370_482_260e03,  # 11
            0,
            +3.908_064_233_624_180e03,  # 13
            0,
            -3.834_967_391_651_310e03,  # 15
            0,
            +2.799_606_547_665_170e03,  # 17
            0,
            -1.512_862_318_866_920e03,  # 19
            0,
            +5.961_601_393_400_090e02,  # 21
            0,
            -1.663_217_393_029_580e02,  # 23
            0,
            +3.109_883_697_398_840e01,  # 25
            0,
            -3.493_493_745_061_900e00,  # 27
            0,
            +1.781_421_569_564_950e-01,  # 29
        ),
    ),
}