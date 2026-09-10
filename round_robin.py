#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections.abc import Iterator, Iterable
from typing import TypeVar
import random
import sys

T = TypeVar('T')
TO = T | None

DATA_DELIM = '---'

def get_matchups(teams: Iterable[T]) -> Iterator[list[tuple[TO, TO]]]:
    """Each yielded value is an iterable of matchups (encoded as a tuple of opposing
    teams) for the round.  Note that `None` in a matchup represents a bye.
    """
    def rotate(my_list: list[TO], n: int = 1) -> list[TO]:
        return my_list[n:] + my_list[:n]

    def matchups(my_field: list[TO]) -> Iterator[tuple[TO, TO]]:
        home = my_field[:n_matchups]
        away = reversed(my_field[n_matchups:])
        return zip(home, away)

    teams_list = list(teams)
    if len(teams_list) % 2 == 1:
        teams_list.append(None)
    random.shuffle(teams_list)
    n_teams = len(teams_list)
    n_matchups = n_teams // 2
    list_head = teams_list[:1]
    list_tail = teams_list[1:]

    for _ in range(n_teams - 1):
        field = list_head + list_tail
        yield matchups(field)
        list_tail = rotate(list_tail)

def main() -> int:
    """Generate a pure round robin bracket, where every team plays every other team
    exactly once.

    Usage: python -m round_robin <nteams>

    Each line in the output represents a round of play.  Each pair of teams (reading from
    left to right, and identified by the numbers `1` through 'nteams`) face each in the
    round.  Byes (if any) are indicated as the odd team at the end of a line.
    """
    usage = lambda x: x + "\n\n" + main.__doc__
    if len(sys.argv) < 2:
        return usage("Number of teams not specified")

    nteams = int(sys.argv[1])
    teams = (str(x) for x in range(1, nteams + 1))
    
    print(DATA_DELIM)
    for round in get_matchups(teams):
        matchups = []
        bye = None
        for matchup in round:
            if None in matchup:
                assert bye is None
                bye = matchup[0] or matchup[1]
                continue
            matchups += matchup
        if bye:
            matchups.append(bye)
        print(','.join(matchups))

    return 0

if __name__ == '__main__':
    sys.exit(main())
