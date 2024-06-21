#!/bin/sh

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

CHECKS="-*, \
    bugprone*, \
    performance*, \
    cppcoreguidelines-*, \
    modernize-*, \
    google-*, \
    misc-*, \
    performance-*, \
    readability-*, \
    clang-analyzer-*, \
    \
    -readability-implicit-bool-conversion, \
    -cppcoreguidelines-pro-bounds-array-to-pointer-decay, \
    -modernize-use-emplace, \
    -modernize-pass-by-value, \
    -modernize-use-auto, \
    -*magic-numbers*, \
    -modernize-use-trailing-return-type, \
    -modernize-use-using, \
    -modernize-use-nodiscard, \
    -modernize-return-braced-init-list"

SOURCE_FILES=$(find ./examples ./src -name "*.cc")
echo $SOURCE_FILES

clang-tidy -p=. --checks="$CHECKS" -header-filter='./src/.*\.h$' $1
