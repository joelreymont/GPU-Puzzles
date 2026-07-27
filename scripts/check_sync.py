#!/usr/bin/env python3
"""Skeleton and solution must be byte-identical outside marked fill-in regions.

Region syntax in both trees:
    /// BEGIN CODE HERE (approx N lines) ///
    ...                                        <- differs between the trees
    /// END CODE HERE ///
Marker lines themselves must match exactly (including the approx count).
"""

import re
import sys
from pathlib import Path

BEGIN = re.compile(r"/// BEGIN CODE HERE \(approx \d+ lines?\) ///")
END = re.compile(r"/// END CODE HERE ///")

ROOT = Path(__file__).resolve().parent.parent


def outside_lines(path: Path) -> list[str]:
    out = []
    inside = False
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if BEGIN.search(line):
            if inside:
                sys.exit(f"{path}:{ln}: nested BEGIN CODE HERE")
            inside = True
            out.append(line)
        elif END.search(line):
            if not inside:
                sys.exit(f"{path}:{ln}: END CODE HERE without BEGIN")
            inside = False
            out.append(line)
        elif not inside:
            out.append(line)
    if inside:
        sys.exit(f"{path}: unterminated CODE HERE region")
    return out


def main() -> int:
    skels = {p.name: p for p in (ROOT / "skeletons").glob("p*")}
    sols = {p.name: p for p in (ROOT / "solutions").glob("p*")}
    if skels.keys() != sols.keys():
        only_sk = sorted(skels.keys() - sols.keys())
        only_so = sorted(sols.keys() - skels.keys())
        sys.exit(f"tree mismatch: skeleton-only={only_sk} solution-only={only_so}")
    if not skels:
        sys.exit("no puzzles found under skeletons/")

    fail = False
    for name in sorted(skels):
        sk = skels[name] / "kernel.cu"
        so = sols[name] / "kernel.cu"
        a, b = outside_lines(sk), outside_lines(so)
        regions = sum(1 for l in a if BEGIN.search(l))
        if regions == 0:
            print(f"DIFF {name}: skeleton has no CODE HERE region")
            fail = True
            continue
        if a != b:
            print(f"DIFF {name}: trees diverge outside fill-in regions")
            for i, (x, y) in enumerate(zip(a, b), 1):
                if x != y:
                    print(f"  first divergence at outside-line {i}:")
                    print(f"    skeleton: {x}")
                    print(f"    solution: {y}")
                    break
            else:
                print(f"  length differs: skeleton={len(a)} solution={len(b)}")
            fail = True
        else:
            print(f"OK   {name} ({regions} region{'s' if regions > 1 else ''})")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
