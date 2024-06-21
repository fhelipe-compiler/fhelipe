#!/usr/bin/env python3

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

import re

from submit import experiments_root

error_re = re.compile(r"Min error: (\d+\.\d*)", flags=re.I)


def get_train_error(log):
    with open(log) as f:
        try:
            last_line = f.readlines()[-1]
            error = float(error_re.search(last_line).groups()[0])
            return error
        except (IndexError, AttributeError, ValueError):
            return float("inf")


def get_results():
    root = experiments_root()
    for train_c in root.iterdir():
        train_id = train_c.name
        for inst_c in train_c.iterdir():
            inst_id = inst_c.name

            path = inst_c / "training" / "log.txt"
            error = get_train_error(path)
            yield (inst_id, train_id, error)


def print_results():
    results = list(get_results())
    results = sorted(results, key=lambda x: x[2])

    template = "{:<20} | {:<50} | {:.2f}"
    print("{:<20} | {:<50} | {}".format("Instance", "Training", "Error %"))

    for ints, train, error in results:
        print(template.format(ints, train, error))


if __name__ == "__main__":
    print_results()
