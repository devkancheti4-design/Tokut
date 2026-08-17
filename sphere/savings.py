# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""What a ruling loop saves. Rates are MEASURED, not estimated:
   1,644 tok/ruling  -- 24,660 tokens for 15 rulings, 8-bit situation, batched
   3,781 tok/ruling  -- 60,491 tokens for 16 rulings, rich state, batched
  15,100 tok/job     -- 60,491 tokens for 4 jobs, full in-loop control
Law C rules at 0 tokens. The saving is rulings x rate. Generation is untouched.
    python3 -m sphere.savings --rulings-per-task 8 --tasks-per-day 40
"""
import argparse
RATES = [("8-bit situation, batched", 1644), ("rich state, batched", 3781)]

a = argparse.ArgumentParser()
a.add_argument("--rulings-per-task", type=float, default=8)
a.add_argument("--tasks-per-day", type=float, default=40)
a.add_argument("--price-per-mtok", type=float, default=15.0)
g = a.parse_args()

r_day = g.rulings_per_task * g.tasks_per_day
print("assumption you supply : %.0f rulings/task x %.0f tasks/day = %.0f rulings/day\n"
      % (g.rulings_per_task, g.tasks_per_day, r_day))
print("%-28s %-14s %-14s %s" % ("if each ruling is a call", "tokens/day", "tokens/month", "$/month"))
print("-" * 74)
for label, rate in RATES:
    d = r_day * rate
    print("%-28s %-14s %-14s $%s"
          % (label, format(int(d), ","), format(int(d * 30), ","),
             format(int(d * 30 / 1e6 * g.price_per_mtok), ",")))
print("%-28s %-14s %-14s %s" % ("with the law ruling", "0", "0", "$0"))
print("\nThis is the saving on RULINGS ONLY. Every token spent generating code,")
print("text or patches is untouched -- the law has no capability there.")
