#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate round robin brackets using Constraint Programming (CP), where higher seeded
teams are guaranteed to have an easier schedule.

To Do:

- Fix the kludgy 0-based vs. 1-based team/round stuff in `validate_bracket`
- Take into account difficulty of opponents in adjacent (or near-adjacent) rounds
"""

from statistics import mean, median
import sys
import os

import numpy as np
from scipy.stats import linregress

from ortools.sat.python import cp_model

DEBUG = int(os.environ.get('ROUND_ROBIN_DEBUG') or 0)

def build_bracket(nteams: int, nrounds: int) -> list | None:
    """Attempt to build a bracket with the specified configuration.  Return ``None`` if
    solver is unable to come up with a solution given the specified parameters and/or
    bounds.

    If the number of teams is odd, then a "ghost" team is created to pair with the bye
    team for each round.

    NOTE: this function currently only works for nrounds < nteams < 2 * nrounds

    Constraints:

    1. Each table seats exactly 2 teams (including the ghost, if exists) in every round

    2. Each team (including the ghost) sits at exactly one table in every round

    3. Pairs of teams may not sit at the same table in more than one round (including the
       ghost)

    4. Additional constraints programmatically generated to generally pair higher seeds
       with lower seeds (ATTN: algorithm still in development!)
    """
    assert nteams > nrounds
    assert nteams <= 2 * nrounds

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

    # Constraint #4 - additional constraints to pair higher seeds with lower seeds--the
    # current approach is to make sure the top seeds all meet the lowest seed (or ghost),
    # and loosen the weakness constraints as we go down through the field
    #
    # REVISIT: this is not close to optimal (in fact, doesn't even generate reliably
    # monotonic results), but at least can be expressed compactly.  We'll use this for
    # now, but definitely need to keep working on a formula that does a better job!!!
    hi = tteams
    lo_range = range(hi - nrounds, hi)
    for t1, lo in enumerate(lo_range):
        for t2 in range(lo, hi):
            if t2 <= t1:
                continue
            model.add(mtgs_map[t1][t2] == 1)

    # Constraint #5a - for top half of the bracket, ensure there are more lower seeded
    # opponents than higher seeded opponents
    pass

    # Constraint #5b - for bottom half of the bracket, ensure there are more higher seeded
    # opponents than lower seeded opponents
    pass

    # Constraint #6 - ensure that the average opponent seed level (across all rounds) goes
    # up monotonically as we walk down the seed ladder
    pass

    # Constraint #7 - optimize for minimum MSE of aggregate opponent stength relative to
    # linear reference
    pass

    solver = cp_model.CpSolver()
    if DEBUG:
        solver.parameters.log_search_progress = True
        if DEBUG > 1:
            solver.parameters.log_subsolver_statistics = True
    status = solver.solve(model)
    print(f"Status: {status} ({solver.status_name(status)})", file=sys.stderr)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    bracket = []
    for r in rounds:
        bracket.append([])
        for b in tables:
            bracket[-1].append([t for t in all_teams if solver.value(seats[(t, r, b)])])

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

    print("\n        Opponent Stats"
          "\n    Min  Max  Median  Mean"
          "\n    ---  ---  ------  -----")
    for i, st in enumerate(opp_stats):
        print(f"{i + 1:2d}: {st[0]:3d}  {st[1]:3d}  {st[2]:6.2f}  {st[3]:5.2f}")

    assert len(opp_stats) == nteams
    opp_data = [st[3] for st in opp_stats]
    lin_act = linregress(range(nteams), opp_data)
    #print(f"lin_act: {lin_act}")
    act_val = lambda x: lin_act.slope * x + lin_act.intercept

    ref_x = [0, nteams - 1]
    ref_y = [opp_data[0], opp_data[-1]]
    lin_ref = linregress(ref_x, ref_y)
    #print(f"lin_ref: {lin_ref}")
    ref_val = lambda x: lin_ref.slope * x + lin_ref.intercept
    ref_func = np.vectorize(ref_val)
    ref_data = ref_func(np.array(range(nteams)))

    mse = ((ref_data - np.array(opp_data)) ** 2).mean()
    rmse = np.sqrt(mse)

    print("\nSlope (for Mean)")
    print(f"- Reference: {lin_ref.slope:.2f}")
    print(f"- Actual:    {lin_act.slope:.2f}")
    print(f"- Diff:      {(lin_ref.slope-lin_act.slope)/lin_ref.slope*100.0:.1f}%")
    print("\nExtrapolated Mean")
    print(f"- Range:     {act_val(0):.2f} - {act_val(nteams-1):.2f}")
    print("\nLinearity")
    print(f"- R-Sqaured: {lin_act.rvalue**2:.2f}")
    print("\nFairness")
    print(f"- RMSE:      {rmse:.2f}")

    # LATER: validate strict ordering of schedule difficulty!!!
    return True

def print_bracket(bracket: list) -> None:
    """Print human-readable representation of the generated backed (internal format).
    """
    for i, round in enumerate(bracket):
        print(f"\nRound {i + 1}:")
        for j, table in enumerate(round):
            print(f"  Table {j + 1}: {[t + 1 for t in table]}")

def print_bracket_csv(bracket: list) -> None:
    """Print CSV for generated bracket, compatible with input format expected by
    ``tourn_eval`` (which doesn't really exist yet!).
    """
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
