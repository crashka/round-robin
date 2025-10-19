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

    model = cp_model.CpModel()

    # Constraint #0 - specify domain for (team, round, table)
    seats = {}
    for p in all_teams:
        for r in rounds:
            for t in tables:
                seats[(p, r, t)] = model.new_bool_var(f'seat_p{p}_r{r}_t{t}')

    # Constraint #1 - for every round, each table seats 2 teams
    for r in rounds:
        for t in tables:
            model.add(sum(seats[(p, r, t)] for p in all_teams) == 2)

    # Constraint #2 - every team sits at one table per round
    for p in all_teams:
        for r in rounds:
            model.add(sum(seats[(p, r, t)] for t in tables) == 1)

    # Constraint #3 - every pair of teams meets no more than once across all rounds
    mtgs_map = {}
    for p1 in all_teams[:-1]:
        mtgs_map[p1] = {}
        for p2 in all_teams[p1+1:]:
            mtgs_map[p1][p2] = []
            mtgs = mtgs_map[p1][p2]
            for r in rounds:
                for t in tables:
                    mtg = model.new_bool_var(f'mtg_p{p1}_p{p2}_r{r}_t{t}')
                    model.add_multiplication_equality(mtg, [seats[(p1, r, t)], seats[(p2, r, t)]])
                    mtgs.append(mtg)
            model.add(sum(mtgs) < 2)

    # Constraint #4 - additional constraints to pair higher seeds with lower seeds--the
    # current approach is to make sure the top seeds all meet the lowest seed (or ghost),
    # and loosen the weakness constraints as we go down through the field
    #
    # REVISIT: this is not close to optimal (in fact, doesn't even generate reliably
    # monotonic results), but at least can be expressed compactly.  We'll use this for
    # now, but definitely need to keep working on a formula that does a better job!!!
    hi = tteams
    lo_range = range(hi - nrounds, hi)
    for p1, lo in enumerate(lo_range):
        for p2 in range(lo, hi):
            if p2 <= p1:
                continue
            mtgs = mtgs_map[p1][p2]
            model.add(sum(mtgs) == 1)

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
        for t in tables:
            bracket[-1].append([p for p in all_teams if solver.value(seats[(p, r, t)])])

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
    all_teams = set(p + 1 for p in range(nteams))

    for round in bracket_in:
        for table in round:
            assert len(table) == 2
            p1, p2 = table
            team_opps[p1].append(p2+1)
            team_opps[p2].append(p1+1)
    assert len(team_opps) == nteams

    opp_stats = []
    print("\nOpponents by Round")

    for p, opps in enumerate(team_opps):
        uniq = set(opps)
        assert len(uniq) == len(opps)
        seed = p + 1
        no_play = all_teams - uniq - set([seed])
        print(f"{seed:2d}: play: {opps}, no play: {sorted(no_play)}")
        opp_stats.append((min(opps), max(opps), median(opps), mean(opps)))

    print("\n        Opponent Stats"
          "\n    Min  Max  Median  Mean"
          "\n    ---  ---  ------  -----")
    for idx, st in enumerate(opp_stats):
        print(f"{idx+1:2d}: {st[0]:3d}  {st[1]:3d}  {st[2]:6.2f}  {st[3]:5.2f}")

    assert len(opp_stats) == nteams
    opp_data = [st[3] for st in opp_stats]
    lin_act = linregress(range(nteams), opp_data)
    #print(f"lin_act: {lin_act}")
    act_val = lambda x: lin_act.slope * x + lin_act.intercept

    ref_x = [0, nteams]
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
    for r, round in enumerate(bracket):
        print(f"\nRound {r+1}:")
        for t, table in enumerate(round):
            print(f"  Table {t+1}: {[p+1 for p in table]}")

def print_bracket_csv(bracket: list) -> None:
    """Print CSV for generated bracket, compatible with input format expected by
    ``tourn_eval`` (which doesn't really exist yet!).
    """
    for round in bracket:
        print(','.join([str(p + 1) for table in round for p in table]))

########
# main #
########

def main() -> int:
    """Usage::

      $ python -m round_robin_cp <nteams> <nrounds> [<csvout>]

    where any non-empty value for ``csvout`` specifies that the bracket should be output
    in CSV format (as expected by ``tourn_eval``); otherwise a human-readable version is
    printed.
    """
    nteams  = int(sys.argv[1])
    nrounds = int(sys.argv[2])
    csvout  = None
    if len(sys.argv) > 3:
        csvout = bool(sys.argv[3])
        if len(sys.argv) > 4:
            print(f"Invalid arg(s): {' '.join(sys.argv[4:])}", file=sys.stderr)
            return 1

    bracket = build_bracket(nteams, nrounds)
    if not bracket:
        print("Unable to build bracket", file=sys.stderr)
        return 1

    if csvout:
        print_bracket_csv(bracket)
    else:
        #print_bracket(bracket)
        pass
    return 0

if __name__ == "__main__":
    sys.exit(main())
