"""Tests for the /failure-probes endpoint and its planning core."""
from __future__ import annotations

import random
from functools import lru_cache

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Node, ProbeInternal, ProbeLeaf, ProbeObservation
from app.planner import PlanError
from app.probes import MAX_CLOSURE_NODES, plan_failure_probes

client = TestClient(app)


# ---------------------------------------------------------------- helpers


def node(name: str, content: str, deps: list[str] | None = None) -> Node:
    return Node(name=name, content_digest=content, dependency_names=deps or [])


def diamond_nodes() -> list[Node]:
    # a -> b, c -> t.
    return [
        node("a", "a"),
        node("b", "b", ["a"]),
        node("c", "c", ["a"]),
        node("t", "t", ["b", "c"]),
    ]


def plan(nodes, target="t", observations=None):
    obs = [ProbeObservation(node=n, result=r) for n, r in (observations or [])]
    return plan_failure_probes(nodes, target, obs)


def closure_set(graph: dict[str, list[str]], root: str) -> frozenset[str]:
    seen: set[str] = set()
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(graph[name])
    return frozenset(seen)


def reference_height(nodes: list[Node], target: str, observations) -> int:
    """Independent frozenset-DP implementation of the same minimax problem."""
    graph = {n.name: list(n.dependency_names) for n in nodes}
    candidates = set(closure_set(graph, target))
    for probe_name, result in observations:
        fail_set = closure_set(graph, probe_name)
        if result == "failed":
            candidates &= fail_set
        else:
            candidates -= fail_set
    candidates = frozenset(candidates)

    @lru_cache(maxsize=None)
    def height(cands: frozenset[str]) -> int:
        if len(cands) <= 1:
            return 0
        best: tuple[int, str] | None = None
        for probe_name in sorted(graph):
            fail_set = closure_set(graph, probe_name)
            failed = cands & fail_set
            passed = cands - fail_set
            if not failed or not passed:
                continue
            value = 1 + max(height(passed), height(failed))
            key = (value, probe_name)
            if best is None or key < best:
                best = key
        assert best is not None
        return best[0]

    return height(candidates)


def simulate(tree, graph: dict[str, list[str]], fault: str) -> str:
    """Walk the tree for a given fault and return the leaf fault name."""
    while isinstance(tree, ProbeInternal):
        fails = fault in closure_set(graph, tree.probe)
        tree = tree.failed if fails else tree.passed
    assert isinstance(tree, ProbeLeaf)
    return tree.fault


def tree_height(tree) -> int:
    if isinstance(tree, ProbeLeaf):
        return 0
    return 1 + max(tree_height(tree.passed), tree_height(tree.failed))


def leaves(tree) -> set[str]:
    if isinstance(tree, ProbeLeaf):
        return {tree.fault}
    return leaves(tree.passed) | leaves(tree.failed)


def candidate_set(nodes: list[Node], target: str, observations) -> set[str]:
    graph = {n.name: list(n.dependency_names) for n in nodes}
    cands = set(closure_set(graph, target))
    for probe_name, result in observations:
        fail_set = closure_set(graph, probe_name)
        if result == "failed":
            cands &= fail_set
        else:
            cands -= fail_set
    return cands


def assert_tree_valid(result, nodes, target, observations):
    graph = {n.name: list(n.dependency_names) for n in nodes}
    cands = candidate_set(nodes, target, observations)
    assert leaves(result.tree) == cands
    assert result.worst_case == tree_height(result.tree)
    assert result.worst_case == reference_height(nodes, target, observations)
    for fault in cands:
        assert simulate(result.tree, graph, fault) == fault
    observed = {n for n, _ in observations}

    def walk(tree):
        if isinstance(tree, ProbeLeaf):
            return
        assert tree.probe not in observed
        walk(tree.passed)
        walk(tree.failed)

    walk(result.tree)


# ------------------------------------------------------------- basic trees


def test_single_node_closure_is_zero_probes():
    result = plan([node("solo", "x")], target="solo")
    assert result.worst_case == 0
    assert isinstance(result.tree, ProbeLeaf)
    assert result.tree.fault == "solo"


def test_chain_binary_search_height():
    # v0 -> v1 -> ... -> v9 (v9 target); prefix probes allow binary search.
    nodes = [node("v0", "x")]
    for i in range(1, 10):
        nodes.append(node(f"v{i}", "x", [f"v{i-1}"]))
    result = plan(nodes, target="v9")
    assert result.worst_case == 4  # ceil(log2(10))
    assert_tree_valid(result, nodes, "v9", [])


def test_diamond_worst_case_and_tree():
    nodes = diamond_nodes()
    result = plan(nodes, target="t")
    assert result.worst_case == 2
    assert_tree_valid(result, nodes, "t", [])
    # Smallest-named optimal probe is b (splits {a,b} vs {c,t}); a would
    # force a 3-probe worst case.
    assert result.tree.probe == "b"


def test_antichain_without_outside_nodes_is_linear():
    nodes = [node(n, n) for n in "abcd"]
    nodes.append(node("t", "t", list("abcd")))
    result = plan(nodes, target="t")
    # Closure probes only yield singleton splits for independent leaves.
    assert result.worst_case == 4
    assert_tree_valid(result, nodes, "t", [])
    assert result.tree.probe == "a"


def test_outside_nodes_may_split_balanced():
    nodes = [node(n, n) for n in "abcd"]
    nodes.append(node("t", "t", list("abcd")))
    nodes.append(node("ab", "ab", ["a", "b"]))
    nodes.append(node("cd", "cd", ["c", "d"]))
    result = plan(nodes, target="t")
    assert result.worst_case == 3
    assert_tree_valid(result, nodes, "t", [])


def test_uninformative_outside_node_changes_nothing():
    nodes = diamond_nodes()
    nodes.append(node("lonely", "l"))  # no path to/from the closure
    result = plan(nodes, target="t")
    baseline = plan(diamond_nodes(), target="t")
    assert result.worst_case == baseline.worst_case
    assert_tree_valid(result, nodes, "t", [])


# ----------------------------------------------------------- observations


def test_passed_observation_narrows_candidates():
    nodes = diamond_nodes()
    result = plan(nodes, observations=[("b", "passed")])
    # b passed rules out a and b; candidates {c, t}.
    assert result.worst_case == 1
    assert_tree_valid(result, nodes, "t", [("b", "passed")])


def test_failed_observation_narrows_candidates():
    nodes = diamond_nodes()
    result = plan(nodes, observations=[("b", "failed")])
    # b failed leaves {a, b}, probed via a.
    assert result.worst_case == 1
    assert result.tree.probe == "a"
    assert_tree_valid(result, nodes, "t", [("b", "failed")])


def test_observations_identify_fault_with_zero_probes():
    nodes = diamond_nodes()
    result = plan(
        nodes,
        observations=[("b", "passed"), ("c", "failed"), ("a", "passed")],
    )
    assert result.worst_case == 0
    assert isinstance(result.tree, ProbeLeaf)
    assert result.tree.fault == "c"


def test_observed_node_is_never_reprobed():
    nodes = diamond_nodes()
    result = plan(nodes, observations=[("a", "failed")])
    assert_tree_valid(result, nodes, "t", [("a", "failed")])

    def probes(tree, acc):
        if isinstance(tree, ProbeLeaf):
            return acc
        acc.add(tree.probe)
        probes(tree.passed, acc)
        probes(tree.failed, acc)
        return acc

    assert "a" not in probes(result.tree, set())


def test_passed_target_is_contradiction():
    with pytest.raises(PlanError) as exc:
        plan(diamond_nodes(), observations=[("t", "passed")])
    assert exc.value.code == "contradictory_observation"


def test_jointly_contradictory_observations():
    nodes = [node("a", "a"), node("b", "b", ["a"]), node("t", "t", ["b"])]
    # a failed and b passed cannot both hold.
    with pytest.raises(PlanError) as exc:
        plan(nodes, observations=[("a", "failed"), ("b", "passed")])
    assert exc.value.code == "contradictory_observation"


def test_duplicate_observation_same_result():
    with pytest.raises(PlanError) as exc:
        plan(diamond_nodes(), observations=[("b", "passed"), ("b", "passed")])
    assert exc.value.code == "duplicate_observation"
    assert exc.value.details == {"nodes": ["b"]}


def test_duplicate_observation_conflicting_result():
    with pytest.raises(PlanError) as exc:
        plan(diamond_nodes(), observations=[("b", "passed"), ("b", "failed")])
    assert exc.value.code == "duplicate_observation"


def test_duplicate_observations_are_sorted_and_order_independent():
    nodes = diamond_nodes()
    obs1 = [("c", "passed"), ("b", "passed"), ("b", "failed"), ("c", "failed")]
    obs2 = list(reversed(obs1))
    with pytest.raises(PlanError) as e1:
        plan(nodes, observations=obs1)
    with pytest.raises(PlanError) as e2:
        plan(nodes, observations=obs2)
    assert e1.value.code == e2.value.code == "duplicate_observation"
    assert e1.value.details == e2.value.details == {"nodes": ["b", "c"]}


def test_outside_passed_observation_is_informative():
    nodes = diamond_nodes()
    nodes.append(node("ab", "ab", ["a", "b"]))
    result = plan(nodes, observations=[("ab", "passed")])
    assert_tree_valid(result, nodes, "t", [("ab", "passed")])
    assert result.worst_case == 1  # candidates {c, t}


def test_independent_outside_failed_observation_contradicts():
    nodes = diamond_nodes()
    nodes.append(node("lonely", "l"))
    with pytest.raises(PlanError) as exc:
        plan(nodes, observations=[("lonely", "failed")])
    assert exc.value.code == "contradictory_observation"


# -------------------------------------------------------------- validation


def test_unknown_target():
    with pytest.raises(PlanError) as exc:
        plan(diamond_nodes(), target="ghost")
    assert exc.value.code == "unknown_target"
    assert exc.value.details == {"target": "ghost"}


def test_unknown_probe_node():
    with pytest.raises(PlanError) as exc:
        plan(diamond_nodes(), observations=[("ghost", "passed")])
    assert exc.value.code == "unknown_probe_node"
    assert exc.value.details == {"names": ["ghost"]}


def test_unknown_probe_node_checked_before_duplicate():
    with pytest.raises(PlanError) as exc:
        plan(
            diamond_nodes(),
            observations=[("ghost", "passed"), ("ghost", "failed")],
        )
    assert exc.value.code == "unknown_probe_node"


def test_closure_limit_is_ten():
    ok = [node(f"n{i}", "x") for i in range(9)]
    ok.append(node("t", "t", [f"n{i}" for i in range(9)]))
    result = plan(ok, target="t")
    assert result.worst_case == 9

    too_big = [node(f"n{i}", "x") for i in range(10)]
    too_big.append(node("t", "t", [f"n{i}" for i in range(10)]))
    with pytest.raises(PlanError) as exc:
        plan(too_big, target="t")
    assert exc.value.code == "closure_too_large"
    assert exc.value.details == {"nodes": 11, "limit": MAX_CLOSURE_NODES}


def test_graph_errors_propagate():
    # Missing dependency.
    bad_dep = [node("t", "t", ["ghost"])]
    with pytest.raises(PlanError) as exc:
        plan(bad_dep, target="t")
    assert exc.value.code == "missing_dependency"

    # Cycle.
    cyc = [node("a", "a", ["b"]), node("b", "b", ["a"])]
    with pytest.raises(PlanError) as exc:
        plan(cyc, target="a")
    assert exc.value.code == "cycle_detected"

    # Duplicate node.
    dup = [node("a", "1"), node("a", "2")]
    with pytest.raises(PlanError) as exc:
        plan(dup, target="a")
    assert exc.value.code == "duplicate_node"


# ------------------------------------------------------------- determinism


def test_payload_order_does_not_matter():
    rng = random.Random(4242)
    base = diamond_nodes()
    # b failed keeps {a, b}; c passed keeps {a, b}; leaves {a, b} non-singleton.
    obs = [("b", "failed"), ("c", "passed")]
    first = plan(base, observations=obs)
    payloads = [first.model_dump_json()]
    for _ in range(10):
        shuffled = base[:]
        rng.shuffle(shuffled)
        shuffled_obs = obs[:]
        rng.shuffle(shuffled_obs)
        other = plan(shuffled, observations=shuffled_obs)
        payloads.append(other.model_dump_json())
    assert len(set(payloads)) == 1


# --------------------------------------------------------------- fuzz / DP


def _random_dag(rng: random.Random) -> tuple[list[Node], str]:
    names = [f"n{i}" for i in range(rng.randint(1, 8))]
    graph: dict[str, list[str]] = {name: [] for name in names}
    for i, name in enumerate(names):
        possible = names[:i]
        chosen = [d for d in possible if rng.random() < 0.35]
        graph[name] = chosen
    target = rng.choice(names)
    # A few outside nodes that may depend on closure members.
    nodes = [
        Node(name=n, content_digest=n, dependency_names=list(graph[n]))
        for n in names
    ]
    for j in range(rng.randint(0, 4)):
        deps = [d for d in names if rng.random() < 0.4]
        nodes.append(Node(name=f"x{j}", content_digest="x", dependency_names=deps))
    return nodes, target


def test_fuzz_matches_reference_and_tree_simulates():
    rng = random.Random(99)
    for _ in range(400):
        nodes, target = _random_dag(rng)
        graph = {n.name: list(n.dependency_names) for n in nodes}
        cands = set(closure_set(graph, target))
        observations = []
        # Pick a random hidden fault and record random consistent probes.
        hidden = rng.choice(sorted(cands))
        for probe_name in sorted(graph):
            if rng.random() < 0.3:
                result = "failed" if hidden in closure_set(graph, probe_name) else "passed"
                observations.append((probe_name, result))
        response = plan(nodes, target=target, observations=observations)
        assert_tree_valid(response, nodes, target, observations)
        assert simulate(response.tree, graph, hidden) == hidden
        if len(candidate_set(nodes, target, observations)) == 1:
            assert response.worst_case == 0


def test_fuzz_contradictions_detected():
    rng = random.Random(7)
    found = 0
    for _ in range(400):
        nodes, target = _random_dag(rng)
        graph = {n.name: list(n.dependency_names) for n in nodes}
        cands = set(closure_set(graph, target))
        observations = [
            (name, rng.choice(["passed", "failed"]))
            for name in sorted(graph)
            if rng.random() < 0.5
        ]
        # Compute what the reference candidate set would be.
        for probe_name, result in observations:
            fail_set = closure_set(graph, probe_name)
            cands = (cands & fail_set) if result == "failed" else (cands - fail_set)
        try:
            response = plan(nodes, target=target, observations=observations)
        except PlanError as exc:
            assert exc.code == "contradictory_observation"
            assert not cands
            found += 1
        else:
            assert cands
            assert_tree_valid(response, nodes, target, observations)
    assert found > 0


# ------------------------------------------------------------------ HTTP


def test_http_endpoint_basic():
    payload = {
        "nodes": [n.model_dump() for n in diamond_nodes()],
        "target": "t",
    }
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["worst_case"] == 2
    assert set(body) == {"worst_case", "tree"}


def test_http_leaf_when_one_candidate():
    payload = {
        "nodes": [n.model_dump() for n in diamond_nodes()],
        "target": "t",
        "observations": [
            {"node": "b", "result": "passed"},
            {"node": "c", "result": "failed"},
            {"node": "a", "result": "passed"},
        ],
    }
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 200
    assert response.json() == {"worst_case": 0, "tree": {"fault": "c"}}


@pytest.mark.parametrize(
    "extra,code",
    [
        ({"target": "ghost"}, "unknown_target"),
        (
            {
                "observations": [{"node": "ghost", "result": "passed"}],
            },
            "unknown_probe_node",
        ),
        (
            {
                "observations": [
                    {"node": "b", "result": "passed"},
                    {"node": "b", "result": "failed"},
                ],
            },
            "duplicate_observation",
        ),
        (
            {
                "observations": [{"node": "t", "result": "passed"}],
            },
            "contradictory_observation",
        ),
    ],
)
def test_http_semantic_errors(extra, code):
    payload = {"nodes": [n.model_dump() for n in diamond_nodes()], "target": "t"}
    payload.update(extra)
    response = client.post("/failure-probes", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == code


def test_http_422_on_unknown_field_and_bad_literal():
    good_nodes = [n.model_dump() for n in diamond_nodes()]
    response = client.post(
        "/failure-probes",
        json={"nodes": good_nodes, "target": "t", "bogus": 1},
    )
    assert response.status_code == 422

    response = client.post(
        "/failure-probes",
        json={
            "nodes": good_nodes,
            "target": "t",
            "observations": [{"node": "b", "result": "maybe"}],
        },
    )
    assert response.status_code == 422

    response = client.post(
        "/failure-probes",
        json={"nodes": good_nodes, "target": ""},
    )
    assert response.status_code == 422


def test_http_observations_default_to_empty():
    response = client.post(
        "/failure-probes",
        json={"nodes": [node("solo", "x").model_dump()], "target": "solo"},
    )
    assert response.status_code == 200
    assert response.json() == {"worst_case": 0, "tree": {"fault": "solo"}}
