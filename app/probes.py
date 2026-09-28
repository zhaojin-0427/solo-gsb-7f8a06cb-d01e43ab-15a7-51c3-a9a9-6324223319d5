"""Single-fault build-failure probe planning.

Given the current build graph (same schema as /plans), a build *target*
that is known to fail, and optionally some probes already completed
(each a node name with its ``passed`` / ``failed`` outcome), build an
adaptive decision tree that locates the unique faulty node with the
minimum possible number of additional probes in the worst case.

Fault model
-----------
Exactly one node in the target's dependency closure (the target and all
of its transitive dependencies) is faulty, and the closure contains at
most ``MAX_CLOSURE_NODES`` (10) nodes.  Probing a node *n* fails iff the
faulty node is *n* itself or one of *n*'s transitive dependencies::

    probe(n) fails  <=>  fault in closure(n)

Since the fault is known to lie in the target closure, probing the target
itself can never split the candidates and is never chosen by the planner.

Observations and candidates
---------------------------
Each observation (n, passed) removes every node in closure(n) from the
candidate set, while (n, failed) keeps exactly the intersection with
closure(n).  A node observed more than once is a ``duplicate_observation``
error (regardless of outcomes); observations that jointly rule out every
closure node are a ``contradictory_observation`` error.  Any graph node
may be probed: a probe outside the target closure that depends on some
closure node yields additional information, while one independent of the
closure is simply uninformative.

Optimal adaptive tree
---------------------
After the observations narrow the candidate set, a minimax dynamic
program over candidate subsets picks, at every state, a probe minimizing
the worst branch height::

    H(S) = 0                              if |S| == 1
    H(S) = 1 + min over probes n of
               max(H(S \\ closure(n)), H(S & closure(n)))

Only probes that actually split S into two non-empty parts are eligible;
such a probe always exists in a DAG closure (the failure sets of distinct
nodes differ).  Among equally good probes the lexicographically smallest
node name is chosen, independently at each state; optimal substructure
makes this equivalent to the lexicographically smallest full tree.  Probes
are grouped by their failure bit-mask (the group representative is its
smallest-named node), since two probes with identical outcomes on the
candidates are interchangeable.

Determinism
-----------
All validation and enumeration goes through name-sorted structures, so
the request payload order (including the observations) never changes the
result.
"""
from __future__ import annotations

from functools import lru_cache

from app.models import (
    FailureProbeResponse,
    Node,
    ProbeInternal,
    ProbeLeaf,
    ProbeObservation,
)
from app.planner import PlanError, validate_graph

MAX_CLOSURE_NODES = 10


def _dependency_closure(graph: dict[str, list[str]], target: str) -> set[str]:
    """Target and every node reachable through direct dependencies."""
    closure: set[str] = set()
    stack = [target]
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        closure.add(name)
        stack.extend(graph[name])
    return closure


def _transitive_dependencies(
    graph: dict[str, list[str]], roots: frozenset[str]
) -> set[str]:
    """Every node reachable through dependency edges from any root (roots incl.)."""
    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        stack.extend(graph[name])
    return reachable


def plan_failure_probes(
    nodes: list[Node],
    target: str,
    observations: list[ProbeObservation],
) -> FailureProbeResponse:
    """Validate the request and compute the optimal adaptive probe tree."""
    graph, _contents, _order = validate_graph(nodes)

    # ---- validation: target resolves --------------------------------------
    if target not in graph:
        raise PlanError(
            "unknown_target",
            f"Target node is not in the graph: {target!r}",
            {"target": target},
        )

    # ---- validation: target closure size limit ----------------------------
    closure = _dependency_closure(graph, target)
    if len(closure) > MAX_CLOSURE_NODES:
        raise PlanError(
            "closure_too_large",
            f"The target closure has {len(closure)} nodes; at most "
            f"{MAX_CLOSURE_NODES} are supported",
            {"nodes": len(closure), "limit": MAX_CLOSURE_NODES},
        )

    # ---- validation: observed probe nodes resolve -------------------------
    unknown_probes = sorted({obs.node for obs in observations if obs.node not in graph})
    if unknown_probes:
        raise PlanError(
            "unknown_probe_node",
            "Observed probes reference nodes that are not in the graph",
            {"names": unknown_probes},
        )

    # ---- validation: a node may be observed at most once ------------------
    seen_obs: set[str] = set()
    duplicate_obs: set[str] = set()
    for obs in observations:
        if obs.node in seen_obs:
            duplicate_obs.add(obs.node)
        seen_obs.add(obs.node)
    if duplicate_obs:
        raise PlanError(
            "duplicate_observation",
            "Each node may be observed at most once",
            {"nodes": sorted(duplicate_obs)},
        )

    # ---- narrow the candidate set with the observations -------------------
    closure_sorted = sorted(closure)
    index = {name: i for i, name in enumerate(closure_sorted)}
    full_mask = (1 << len(closure_sorted)) - 1

    candidate_mask = full_mask
    for obs in observations:
        failure_set = _transitive_dependencies(graph, frozenset({obs.node}))
        fail_mask = 0
        for name in failure_set & closure:
            fail_mask |= 1 << index[name]
        if obs.result == "failed":
            candidate_mask &= fail_mask
        else:
            candidate_mask &= full_mask ^ fail_mask

    if candidate_mask == 0:
        raise PlanError(
            "contradictory_observation",
            "The completed probes rule out every closure node; exactly one "
            "faulty node was assumed",
            {"target": target},
        )

    # ---- failure bit-mask of every graph node (only masks that can split) --
    # Group nodes by identical mask; the smallest-named node of a group is the
    # only probe that can win the lexicographic tie-break for that outcome.
    representatives: dict[int, str] = {}
    for name in sorted(graph):
        failure_set = _transitive_dependencies(graph, frozenset({name}))
        mask = 0
        for member in failure_set & closure:
            mask |= 1 << index[member]
        current = representatives.get(mask)
        if current is None or name < current:
            representatives[mask] = name
    # Mask 0 (probes independent of the closure) and the full closure mask
    # (the target itself) never split a candidate subset; drop both.
    masks = [
        mask
        for mask in representatives
        if mask != 0 and mask != full_mask
    ]
    masks.sort(key=lambda mask: representatives[mask])

    # ---- minimax DP over candidate subsets ---------------------------------
    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, object]:
        """Return (worst-case additional probes, decision tree) for a mask."""
        if mask & (mask - 1) == 0:  # exactly one bit set
            name = closure_sorted[mask.bit_length() - 1]
            return 0, ProbeLeaf(fault=name)

        # (height, probe name, fail mask) of the best probe seen so far.
        best: tuple[int, str, int] | None = None
        for fail_mask in masks:
            failed = mask & fail_mask
            if failed == 0 or failed == mask:
                continue  # probe never fails / always fails on these candidates
            passed = mask ^ failed
            height = 1 + max(solve(passed)[0], solve(failed)[0])
            probe_name = representatives[fail_mask]
            key = (height, probe_name)
            if best is None or key < best[:2]:
                best = (height, probe_name, fail_mask)

        # Unreachable: distinct closure nodes have distinct failure sets, so
        # any subset with >= 2 members admits a splitting probe.
        assert best is not None
        height, probe_name, fail_mask = best
        failed = mask & fail_mask
        passed = mask ^ failed
        _, passed_tree = solve(passed)
        _, failed_tree = solve(failed)
        return height, ProbeInternal(
            probe=probe_name, passed=passed_tree, failed=failed_tree
        )

    worst_case, tree = solve(candidate_mask)
    return FailureProbeResponse(worst_case=worst_case, tree=tree)


__all__ = ["MAX_CLOSURE_NODES", "plan_failure_probes"]
