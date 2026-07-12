#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate round robin brackets using Constraint Programming (CP), where higher seeded
teams are guaranteed to have an easier schedule.

To Do:

- Fix the kludgy 0-based vs. 1-based team/round stuff in `validate_bracket`
- Take into account difficulty of opponents in adjacent (or near-adjacent) rounds
"""

from collections.abc import Iterable
from statistics import mean, median
from math import sqrt
import sys
import os

import numpy as np
from scipy.stats import linregress
from scipy.stats._stats_py import LinregressResult

from ortools.sat.python import cp_model

# Constants
DFLT_PREC_MULT = 10  # appears to work well

# Type aliases
NDArray      = np.typing.NDArray
simple_sum   = cp_model.LinearExpr.sum
weighted_sum = cp_model.LinearExpr.weighted_sum

DEBUG        = int(os.environ.get('CP_DEBUG') or 0)
MAX_TIME     = int(os.environ.get('CP_MAX_TIME') or 0)
PREC_MULT    = int(os.environ.get('CP_PREC_MULT') or DFLT_PREC_MULT)
SKIP_CONSTR  = os.environ.get('CP_SKIP_CONSTR')

class MeanLinRef:
    """Reference linear equation for mean opponent seeds
    """
    nteams:  int
    nrounds: int
    offset:  int
    lin_res: LinregressResult

    def __init__(self, nteams: int, nrounds: int, offset: int = 0):
        """`offset` represents number for highest ranked seed (typically 0 or 1)
        """
        min_y = sum(range(0, nrounds)) / nrounds + offset
        max_y = sum(range(nteams - nrounds, nteams)) / nrounds + offset
        ref_x = [0, nteams - 1]
        ref_y = [max_y, min_y]
        lin_res = linregress(ref_x, ref_y)

        self.nteams  = nteams
        self.nrounds = nrounds
        self.offset  = offset
        self.lin_res = lin_res

    @property
    def slope(self) -> float:
        return self.lin_res.slope

    @property
    def intercept(self) -> float:
        return self.lin_res.intercept

    def y_vals(self, x_vals: Iterable[float]) -> NDArray:
        ref_val = lambda x: self.slope * x + self.intercept
        ref_func = np.vectorize(ref_val)
        return ref_func(np.array(x_vals))

class MeanLinTgt(MeanLinRef):
    """Target linear equation for mean opponent seeds (based on endpoints)
    """
    y_range: tuple[int, int]

    def __init__(self, nteams: int, y_range: tuple[int, int]):
        ref_x = [0, nteams - 1]
        ref_y = list(y_range)
        lin_res = linregress(ref_x, ref_y)

        self.nteams  = nteams
        self.y_range = y_range
        self.lin_res = lin_res

def build_bracket(nteams: int, nrounds: int) -> list | None:
    """Attempt to build a bracket with the specified configuration.  Bracket output format
    is a list (rounds) of lists (tables) of 2-tuple (teams).  Returns ``None`` if solver
    is unable to come up with a solution given the specified parameters and/or bounds.

    If the number of teams is odd, then a "ghost" team is created to pair with the bye
    team for each round.

    NOTE: this function currently only works for nrounds < nteams (see constraint #3)

    Constraints:

    1. Each table seats exactly 2 teams (including the ghost, if exists) in every round

    2. Each team (including the ghost) sits at exactly one table in every round

    3. Pairs of teams may not sit at the same table in more than one round (including the
       ghost)

    4. Ensure that top teams get the byes (if any)

    5. Ensure that more top-half teams face more lower-ranked opponents, and vice versa
    for bottom-half teams

    6. Ensure that strength of schedule monotonically decreases as we walk down the seed
    ladder

    7. Optimize for minimum MSE of aggregate opponent stength relative to linear reference

    """
    assert nteams > nrounds

    nopps   = nteams - 1
    ntables = nopps // 2 + 1
    nghosts = ntables * 2 - nteams
    tteams  = nteams + nghosts
    assert nghosts <= 1
    assert tteams & 0x01 == 0

    teams     = range(nteams)
    rounds    = range(nrounds)
    tables    = range(ntables)
    all_teams = range(tteams)  # includes ghosts
    ghosts    = all_teams[nteams:]
    assert len(ghosts) == nghosts

    model = cp_model.CpModel()
    skip_constr = SKIP_CONSTR.split(',') if SKIP_CONSTR else []
    if skip_constr:
        print(f"Skipping constraints: {skip_constr}", file=sys.stderr)

    # Constraint #0 - specify domain for (team, round, table)
    seats = {}
    for t in all_teams:
        for r in rounds:
            for b in tables:
                seats[(t, r, b)] = model.new_bool_var(f'seat_t{t}_r{r}_b{b}')

    # Build variables and maps related to meetings
    mtgs_map = {t1: {t2: None for t2 in all_teams} for t1 in all_teams}
    for t1 in all_teams[:-1]:
        for t2 in all_teams[t1 + 1:]:
            assert t2 > t1
            assert mtgs_map[t1][t2] is None
            mtgs = []
            for r in rounds:
                for b in tables:
                    mtg = model.new_bool_var(f'mtg_t{t1}_t{t2}_r{r}_b{b}')
                    model.add_multiplication_equality(mtg, [seats[(t1, r, b)], seats[(t2, r, b)]])
                    mtgs.append(mtg)
            mtgs_map[t1][t2] = sum(mtgs)

    # Validate and create inverse mappings
    for t1 in all_teams:
        assert mtgs_map[t1][t1] is None
        for t2 in all_teams[:t1]:
            assert t2 < t1
            assert mtgs_map[t1][t2] is None
            assert mtgs_map[t2][t1].num_exprs == nrounds * ntables
            mtgs_map[t1][t2] = mtgs_map[t2][t1]

    # Build map representing strength of schedule for each team
    sched_strgth = {t: None for t in all_teams}
    for t1 in all_teams:
        assert sched_strgth[t1] is None
        opp_strgth = []
        for t2 in all_teams:
            if t2 == t1:
                continue
            # NOTE: we have to add 1 here to avoid a type error (the backend is too clever
            # in presumptively creating constant linear expressions for zeros, which are
            # then incompatible with the sum() operation)--IMPORTANT: the offset used in
            # constructing `lin_ref` (below) must accord with this number!!!
            opp_strgth.append(mtgs_map[t1][t2] * (t2 + 1))
        sched_strgth[t1] = sum(opp_strgth)

    # Constraint #1 - for every round, each table seats 2 teams
    for r in rounds:
        for b in tables:
            model.add(sum(seats[(t, r, b)] for t in all_teams) == 2)

    # Constraint #2 - every team sits at one table per round
    for t in all_teams:
        for r in rounds:
            model.add(sum(seats[(t, r, b)] for b in tables) == 1)

    # Constraint #3 - every pair of teams meets no more than once across all rounds
    for t1 in all_teams[:-1]:
        for t2 in all_teams[t1 + 1:]:
            assert t2 > t1
            model.add(mtgs_map[t1][t2] < 2)

    # Constraint #4 - if there are byes, ensure that the top seeds get them
    if ghosts:
        assert len(ghosts) == 1
        g = ghosts[0]
        for t in teams[:nrounds]:
            model.add(mtgs_map[t][g] == 1)

    # Constraint #5 - for top-half teams, ensure that at least half of opponents are lower
    # in rank; and vice versa for bottom-half teams
    if '5' not in skip_constr:
        halfway = tteams // 2
        for t1 in all_teams[:halfway]:
            lo_teams = all_teams[t1 + 1:]
            hi_teams = all_teams[:t1]
            lo_seeds = sum(mtgs_map[t1][t2] for t2 in lo_teams)
            hi_seeds = sum(mtgs_map[t1][t2] for t2 in hi_teams)
            model.add(lo_seeds >= hi_seeds)

        for t1 in all_teams[halfway:]:
            lo_teams = all_teams[t1 + 1:]
            hi_teams = all_teams[:t1]
            lo_seeds = sum(mtgs_map[t1][t2] for t2 in lo_teams)
            hi_seeds = sum(mtgs_map[t1][t2] for t2 in hi_teams)
            model.add(hi_seeds >= lo_seeds)

    # Constraint #6 - ensure that the average opponent seed level (across all rounds) goes
    # up (strictly) monotonically as we walk down the seed ladder
    if '6' not in skip_constr:
        for t in all_teams[:-1]:
            model.add(sched_strgth[t] > sched_strgth[t + 1])

    # Constraint #7 - optimize for measured linearity (in addition to minimum MSE of
    # aggregate opponent stength relative to an ideal linear reference)

    # Error (and hence RMSE) calculations need higher resolution than accorded by integer
    # math, so we use a multiplier to get the precision we need
    mult = PREC_MULT

    lin_ref  = MeanLinRef(tteams, nrounds, offset=1)
    ref_data = lin_ref.y_vals(range(tteams))
    ref_max  = round(ref_data[0] * nrounds)
    errs     = []
    err_sqs  = []
    ref_vals = []

    if '7' not in skip_constr:
        # create "target" linear reference between actual endpoints
        tgt_first   = sched_strgth[0]
        tgt_last    = sched_strgth[tteams - 1]
        tgt_slope   = model.new_int_var(-ref_max, 0, f'tgt_slope')
        # note: we are *not* dividing by `tteams - 1`, so representation of slope will be
        # high by a multiplicative factor of `slope_mult`
        model.add(tgt_slope == tgt_last - tgt_first)
        slope_mult  = tteams - 1
        tgt_intcpt  = tgt_first * slope_mult          # normalize to slope
        tgt_errs    = []
        tgt_err_sqs = []
        tgt_vals    = []

    for t in all_teams:
        # convert mean ref_data to ideal value for the sum
        ref_val = round(ref_data[t] * nrounds * mult)
        # compute err^2 against ideal linear reference
        err = model.new_int_var(-ref_max * mult, ref_max * mult, f'err{t}')
        err_sq = model.new_int_var(0, (ref_max * mult) ** 2, f'err_sq{t}')
        model.add(err == sched_strgth[t] * mult - ref_val)
        model.add_multiplication_equality(err_sq, [err, err])
        errs.append(err)
        err_sqs.append(err_sq)
        ref_vals.append(ref_val)

        if '7' not in skip_constr:
            # compute linear target value for the sum
            tgt_num = model.new_int_var(0, ref_max * slope_mult * mult, f'tgt_num{t}')
            tgt_val = model.new_int_var(0, ref_max * mult, f'tgt_val{t}')
            model.add(tgt_num == tgt_slope * t * mult + tgt_intcpt * mult)
            model.add_division_equality(tgt_val, tgt_num, slope_mult)
            # compute err^2 against linear target reference
            tgt_err = model.new_int_var(-ref_max * mult, ref_max * mult, f'tgt_err{t}')
            tgt_err_sq = model.new_int_var(0, (ref_max * mult) ** 2, f'tgt_err_sq{t}')
            model.add(tgt_err == sched_strgth[t] * mult - tgt_val)
            model.add_multiplication_equality(tgt_err_sq, [tgt_err, tgt_err])
            tgt_errs.append(tgt_err)
            tgt_err_sqs.append(tgt_err_sq)
            tgt_vals.append(tgt_val)

    if '7' not in skip_constr:
        model.minimize(sum(err_sqs + tgt_err_sqs))
    else:
        model.minimize(sum(err_sqs))

    validation = model.validate()
    if validation:
        print(f"Validation Error: {validation}", file=sys.stderr)

    solver = cp_model.CpSolver()
    if DEBUG:
        solver.parameters.log_search_progress = True
        if DEBUG > 1:
            solver.parameters.log_subsolver_statistics = True
    if MAX_TIME:
        solver.parameters.max_time_in_seconds = MAX_TIME
    status = solver.solve(model)
    print(f"Status: {status} ({solver.status_name(status)})", file=sys.stderr)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    obj_val = solver.objective_value

    v = solver.value
    err_sq_sum = 0
    for t in all_teams:
        err_sq_sum += v(err_sqs[t])
        # sched_strgth and ref_vals have multiplers of (mult * nrounds)
        print(f"{v(sched_strgth[t]) * mult:5d} {ref_vals[t]:5d} {v(errs[t]):5d} "
              f"{v(err_sqs[t]):5d}", file=sys.stderr)
    err_sq_norm = err_sq_sum / nrounds ** 2 / mult ** 2
    rmse = sqrt(err_sq_norm / tteams)
    print(f"MSE: {err_sq_norm:.3f}, RMSE: {rmse:.3f}", file=sys.stderr)

    if '7' not in skip_constr:
        err_sq_sum = 0
        for t in all_teams:
            err_sq_sum += v(tgt_err_sqs[t])
            # sched_strgth and ref_vals have multiplers of (mult * nrounds)
            print(f"{v(sched_strgth[t]) * mult:5d} {v(tgt_vals[t]):5d} {v(tgt_errs[t]):5d} "
                  f"{v(tgt_err_sqs[t]):5d}", file=sys.stderr)
        err_sq_norm = err_sq_sum / nrounds ** 2 / mult ** 2
        rmse = sqrt(err_sq_norm / tteams)
        print(f"MSE: {err_sq_norm:.3f}, RMSE: {rmse:.3f}", file=sys.stderr)

    bracket = []
    for r in rounds:
        bracket.append([])
        for b in tables:
            seating = tuple(t for t in all_teams if solver.value(seats[(t, r, b)]))
            bracket[-1].append(seating)

    if not validate_bracket(bracket, tteams, nrounds):
        raise RuntimeError("Generated bracket fails validation")

    print("\nSolver Stats", file=sys.stderr)
    print(f"- Conflicts: {solver.num_conflicts}", file=sys.stderr)
    print(f"- Branches:  {solver.num_branches}", file=sys.stderr)
    print(f"- Wall time: {solver.wall_time:.2f} secs", file=sys.stderr)
    return bracket

def validate_bracket(bracket_in: list, nteams: int, nrounds: int) -> bool:
    """Return ``True`` if bracket is correct (i.e. all intended constraints met),
    ``False`` otherwise.
    """
    assert nrounds == len(bracket_in)

    # NOTE: index is zero-based, but inner list values are 1-based (this is pretty
    # ugly--should really FIX!!!)
    team_opps = [[] for _ in range(nteams)]
    all_teams = set(t + 1 for t in range(nteams))

    for round in bracket_in:
        for table in round:
            assert len(table) == 2
            t1, t2 = table
            team_opps[t1].append(t2 + 1)
            team_opps[t2].append(t1 + 1)
    assert len(team_opps) == nteams

    opp_stats = []
    print("\nOpponents by Round")

    for i, opps in enumerate(team_opps):
        uniq = set(opps)
        assert len(uniq) == len(opps)
        seed = i + 1
        no_play = all_teams - uniq - set([seed])
        print(f"{seed:2d}: play: {opps}, no play: {sorted(no_play)}")
        opp_stats.append((min(opps), max(opps), median(opps), mean(opps)))

    lin_ref = MeanLinRef(nteams, nrounds, offset=1)
    ref_data = lin_ref.y_vals(range(nteams))
    ref_err_sq_sum = 0.0

    tgt_range = (opp_stats[0][3], opp_stats[-1][3])
    lin_tgt = MeanLinTgt(nteams, tgt_range)
    tgt_data = lin_tgt.y_vals(range(nteams))
    tgt_err_sq_sum = 0.0

    print("\n                      Opponent Stats"
          "\n    Min  Max  Median  Mean    Ref   R-Err   Tgt   T-Err"
          "\n    ---  ---  ------  -----  -----  -----  -----  -----")
    for i, st in enumerate(opp_stats):
        act_val = st[3]
        ref_val = float(ref_data[i])
        ref_err = act_val - ref_val
        ref_err_sq_sum += ref_err * ref_err
        tgt_val = float(tgt_data[i])
        tgt_err = act_val - tgt_val
        tgt_err_sq_sum += tgt_err * tgt_err
        print(f"{i+1:2d}  {st[0]:3d}  {st[1]:3d}  {st[2]:6.2f}  {st[3]:5.2f}  "
              f"{ref_val:5.2f}  {ref_err:5.2f}  {tgt_val:5.2f}  {tgt_err:5.2f}")

    assert len(opp_stats) == nteams
    opp_data = np.array([st[3] for st in opp_stats])
    lin_regr = linregress(range(nteams), opp_data)
    slope_diff_ref = (lin_regr.slope - lin_ref.slope) / lin_ref.slope * 100.0
    slope_diff_tgt = (lin_regr.slope - lin_tgt.slope) / lin_tgt.slope * 100.0
    regr_val = lambda x: lin_regr.slope * x + lin_regr.intercept

    ref_mse = ((opp_data - ref_data) ** 2).mean()
    ref_rmse = np.sqrt(ref_mse)
    tgt_mse = ((opp_data - tgt_data) ** 2).mean()
    tgt_rmse = np.sqrt(tgt_mse)

    print("\nFairness (vs Ref)")
    print(f"- MSE:       {ref_err_sq_sum:.3f}")
    print(f"- RMSE:      {ref_rmse:.3f}")

    print("\nFitness (vs Tgt)")
    print(f"- MSE:       {tgt_err_sq_sum:.3f}")
    print(f"- RMSE:      {tgt_rmse:.3f}")

    print("\nRegression Stats")
    print(f"- Slope:     {lin_regr.slope:.3f}")
    print(f"  - vs Ref:  {slope_diff_ref:.1f}% (value: {lin_ref.slope:.3f})")
    print(f"  - vs Tgt:  {slope_diff_tgt:.1f}% (value: {lin_tgt.slope:.3f})")
    print(f"- Opps Min:  {regr_val(0):.2f}")
    print(f"- Opps Max:  {regr_val(nteams-1):.2f}")
    print(f"- R-Squared: {lin_regr.rvalue ** 2:.3f}")

    # LATER: validate strict ordering of schedule difficulty!!!
    return True

def print_bracket(bracket: list) -> None:
    """Print human-readable representation of the generated backed (internal format).
    """
    print("")
    for i, round in enumerate(bracket):
        print(f"\nRound {i + 1}:")
        for j, table in enumerate(round):
            print(f"  Table {j + 1}: {[t + 1 for t in table]}")

def print_bracket_csv(bracket: list) -> None:
    """Print CSV for generated bracket, compatible with input format expected by
    ``tourn_eval`` (which doesn't really exist yet!).
    """
    print("\n---")
    for round in bracket:
        print(','.join([str(t + 1) for table in round for t in table]))

########
# main #
########

def main() -> int:
    """Usage::

      $ python -m round_robin_cp <nteams> <nrounds> [<brcktout>]

    where any non-empty value for ``brcktout`` specifies the output format for the bracket
    (i.e. 'cvs' or 'human').
    """
    nteams  = int(sys.argv[1])
    nrounds = int(sys.argv[2])
    brcktout  = None
    if len(sys.argv) > 3:
        brcktout = sys.argv[3]
        if len(sys.argv) > 4:
            print(f"Invalid arg(s): {' '.join(sys.argv[4:])}", file=sys.stderr)
            return 1

    bracket = build_bracket(nteams, nrounds)
    if not bracket:
        print("Unable to build bracket", file=sys.stderr)
        return 1

    if not bool(brcktout):
        pass
    elif brcktout.lower() == 'csv':
        print_bracket_csv(bracket)
    else:
        print_bracket(bracket)
    return 0

if __name__ == "__main__":
    sys.exit(main())
